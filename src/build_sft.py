import random
import argparse
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import tiktoken
import ujson as json
from pydantic import BaseModel
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

from tools import all_tools
from prompt.agent import two_stage_1_prompt, two_stage_2_prompt, qwen3_tool_template


class BuildSFT(BaseModel):
    args: Any

    enc: Any = tiktoken.get_encoding("cl100k_base")
    toolmap: Dict[str, Any] = {toolclass().name: toolclass() for toolclass in all_tools}

    def build_sft(self):
        # Check format
        question_ids = set()
        counter = Counter()
        input_tokens_list = []
        output_tokens_list = []
        sft_data = []

        for rs_file in self.args.rs_files.split(","):
            with open(rs_file, "r") as fin:
                for line in tqdm(fin, desc=f"Building SFT data for {rs_file}: "):
                    item = json.loads(line.strip())

                    if not item:
                        continue

                    question_id = item[0]["extra_info"]["question_id"]
                    if question_id in question_ids:
                        continue

                    counter["total_trajectory"] += 1

                    flag = False
                    for step in item:
                        counter["total_step"] += 1
                        prompt = step["prompt"]
                        if prompt[0]["role"] == "system":
                            system_prompt = prompt[0]["content"]
                            if (
                                "Given a product search query, retrieve the user's relevant memories (dialogue history) and identify their purchase preferences from them."
                                in system_prompt
                            ):
                                system_prompt = two_stage_1_prompt.format(
                                    available_tools=qwen3_tool_template.format(
                                        available_tools="\n".join(
                                            [
                                                tool.to_qwen3_string()
                                                for tool in self.toolmap.values()
                                            ]
                                        )
                                    )
                                )
                            elif (
                                "Given the product search query and the user's purchase preferences, find products or product bundles that exactly match them."
                                in system_prompt
                            ):
                                system_prompt = two_stage_2_prompt.format(
                                    available_tools=qwen3_tool_template.format(
                                        available_tools="\n".join(
                                            [
                                                tool.to_qwen3_string()
                                                for tool in self.toolmap.values()
                                            ]
                                        )
                                    )
                                )
                            else:
                                raise ValueError(
                                    f"Invalid system prompt: {system_prompt}"
                                )
                            prompt[0]["content"] = system_prompt.strip()

                        ground_truth = step["reward_model"]["ground_truth"]

                        check_think_and_tool_call = self._check_think_and_tool_call(
                            ground_truth
                        )
                        check_answer = self._check_answer(ground_truth)

                        if not check_think_and_tool_call and not check_answer:
                            continue

                        data, input_tokens, output_tokens = self._apply_chat_template(
                            prompt, ground_truth
                        )

                        if input_tokens + output_tokens >= self.args.cutoff_len:
                            continue

                        sft_data.append(data)
                        input_tokens_list.append(input_tokens)
                        output_tokens_list.append(output_tokens)
                        counter["success_step"] += 1
                        flag = True
                    if flag:
                        counter["success_trajectory"] += 1
                        question_ids.add(question_id)

        random.shuffle(sft_data)

        # Save in the appropriate format based on file extension
        if self.args.sft_file.endswith(".jsonl"):
            # Save as JSONL format (one JSON object per line)
            with open(self.args.sft_file, "w") as fout:
                for item in sft_data:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            # Save as single JSON array
            with open(self.args.sft_file, "w") as fout:
                json.dump(sft_data, fout, ensure_ascii=False, indent=4)

        print(f"Total trajectory: {counter['total_trajectory']}")
        print(
            f"Success trajectory rate: {counter['success_trajectory'] / counter['total_trajectory']:.3f}"
        )
        print(f"Total step: {counter['total_step']}")
        print(
            f"Success step rate: {counter['success_step'] / counter['total_step']:.3f}"
        )
        print(
            f"Input tokens: avg = {np.mean(input_tokens_list):.3f}, min = {np.min(input_tokens_list):.3f}, max = {np.max(input_tokens_list):.3f}, p = {np.percentile(input_tokens_list, list(range(50, 100, 10)))}"
        )
        print(
            f"Output tokens: avg = {np.mean(output_tokens_list):.3f}, min = {np.min(output_tokens_list):.3f}, max = {np.max(output_tokens_list):.3f}, p = {np.percentile(output_tokens_list, list(range(50, 100, 10)))}"
        )

    def _check_think_and_tool_call(self, content: str) -> bool:
        if not content:
            return False

        content = content.strip()

        for field in ["<think>", "</think>", "<tool_call>", "</tool_call>"]:
            if content.count(field) != 1:
                return False

        if content.index("<think>") >= content.index("</think>"):
            return False

        if content.index("<tool_call>") >= content.index("</tool_call>"):
            return False

        if content.index("<think>") != 0 or content.index("</tool_call>") != len(
            content
        ) - len("</tool_call>"):
            return False

        think = content.split("<think>")[1].split("</think>")[0].strip()
        tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0].strip()

        if not think or not tool_call:
            return False

        # check tool call argments
        try:
            for jsonstr in tool_call.strip().split("\n"):
                jsonobj = json.loads(jsonstr.strip())
                if not jsonobj:
                    return False
                name = jsonobj["name"]
                arguments = jsonobj["arguments"]

                assert isinstance(name, str)
                assert name in self.toolmap
                assert isinstance(arguments, dict)
        except Exception as e:
            return False

        return True

    def _check_answer(self, content: str) -> bool:
        if not content:
            return False

        content = content.strip()

        for field in ["<answer>", "</answer>"]:
            if content.count(field) != 1:
                return False

        if content.index("<answer>") >= content.index("</answer>"):
            return False

        if content.index("<answer>") != 0 or content.index("</answer>") != len(
            content
        ) - len("</answer>"):
            return False

        answer = content.split("<answer>")[1].split("</answer>")[0].strip()

        if not answer:
            return False

        return True

    def _apply_chat_template(
        self, messages: List[Dict[str, str]], ground_truth: str
    ) -> Tuple[Dict[str, str], int, int]:
        """
        Convert messages to LLaMA-Factory Alpaca format.

        Alpaca format structure:
        {
            "instruction": str,  # The task instruction (last user message in the conversation)
            "input": str,        # Additional input context (usually empty)
            "output": str,       # The expected response (ground_truth)
            "system": str,       # Optional system prompt
            "history": list      # Optional conversation history as [["user", "assistant"], ...]
        }
        """
        # Extract system message and conversation history
        # The last user message is the instruction, all previous messages are history
        system = ""
        history = []
        instruction = ""

        # Separate system message from conversation messages
        conversation_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                conversation_msgs.append(msg)

        # The last user message is the instruction
        # All previous user-assistant pairs are history
        if conversation_msgs:
            # Find the last user message
            last_user_idx = -1
            for i in range(len(conversation_msgs) - 1, -1, -1):
                if conversation_msgs[i]["role"] == "user":
                    last_user_idx = i
                    instruction = conversation_msgs[i]["content"]
                    break

            # Build history from messages before the last user message
            temp_history = []
            for i in range(last_user_idx):
                msg = conversation_msgs[i]
                if msg["role"] == "user":
                    temp_history.append([msg["content"], ""])
                elif msg["role"] == "assistant":
                    if temp_history and not temp_history[-1][1]:
                        temp_history[-1][1] = msg["content"]

            # Only keep complete user-assistant pairs
            history = [h for h in temp_history if h[0] and h[1]]

        # Build Alpaca format data
        data = {
            "instruction": instruction,
            "input": "",  # Alpaca format uses empty string for input when not needed
            "output": ground_truth,
        }

        # Add optional fields
        if system:
            data["system"] = system

        if history:
            data["history"] = history

        # Calculate tokens for statistics
        full_content = system + "\n" + instruction
        for h in history:
            full_content += "\n" + h[0] + "\n" + h[1]

        input_tokens = len(self.enc.encode(full_content))
        output_tokens = len(self.enc.encode(ground_truth))

        return data, input_tokens, output_tokens


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--rs_files", type=str, required=True)
    args.add_argument("--sft_file", type=str, required=True)
    args.add_argument("--cutoff_len", type=int, default=16384)
    args = args.parse_args()

    random.seed(42)

    builder = BuildSFT(args=args)
    builder.build_sft()
