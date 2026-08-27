import os
import sys
import re
import time
import copy
from typing import Dict, Any, List, Any, Optional
from queue import Queue

import ujson as json
import portalocker
from tqdm import tqdm
from pydantic import BaseModel

from tools import all_tools
from util.llm import ParallelOpenAICompletion, CompletionRequest
from mem.retriever import ConversationDTO


class AgentLoop(BaseModel):
    conversation_id: str
    conversation: ConversationDTO
    model: str

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_steps: int = 20
    cur_step: int = 0
    toolmap: dict[str, Any] = {toolclass().name: toolclass() for toolclass in all_tools}
    messages: list[Dict[str, str]] = []
    tracker: List[Dict[str, Any]] = []
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def clear_messages(self):
        self.messages.clear()

    def think(self) -> str:
        client = ParallelOpenAICompletion(
            base_url=self.base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
            max_workers=1,
            max_retries=10,
            retry_delay=3,
        )

        kwargs = {
            "messages": self.messages,
            "model": self.model,
            "extra_kwargs": {"extra_headers": {"Accept": "text/event-stream"}},
        }
        if self.model.startswith("gpt-5"):
            kwargs["temperature"] = None
            kwargs["max_completion_tokens"] = 8192
        else:
            kwargs["temperature"] = 0.0
            kwargs["max_tokens"] = 8192

        llm_request = CompletionRequest(**kwargs)
        result = client.batch_complete([llm_request])[0]
        if result.success:
            self.prompt_tokens = result.prompt_tokens
            self.completion_tokens = result.completion_tokens
            reasoning_content = result.reasoning_content
            content = result.content

            if reasoning_content:
                reasoning_content = (
                    reasoning_content.strip()
                    .replace("<think>", "")
                    .replace("</think>", "")
                )
                reasoning_content = f"<think>{reasoning_content}</think>"
                content = f"{reasoning_content}\n{content}"
            return content
        else:
            print(f"Think Error: {result.error}", file=sys.stderr)
            return ""

    def act(self, content: str) -> str:
        tool_responses = []

        # Use findall to extract all <tool_call>...</tool_call> blocks
        tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)

        tool_calls = []
        for tool_call_block in tool_call_blocks:
            tool_call_block = tool_call_block.strip()
            for tool_call in tool_call_block.split("\n"):
                tool_call = tool_call.strip()
                if not tool_call:
                    continue
                tool_calls.append(tool_call)

        for tool_call in tool_calls:
            try:
                jsonobj = json.loads(tool_call)
                name = jsonobj["name"]
                arguments = jsonobj.get("arguments")
                if not arguments or not isinstance(arguments, dict):
                    return f"The arguments of tool call is invalid: {tool_call}"
            except Exception as e:
                return f"Parsing tool call failed: {e}"

            if name not in self.toolmap:
                return f"Invalid tool name: {name}"

            tool = self.toolmap[name]
            arguments["conversation_id"] = self.conversation_id
            tool_response = tool.execute(**arguments)

            tool_responses.append(tool_response)

        tool_responses_str = "\n\n".join(tool_responses)
        return f"<tool_response>\n{tool_responses_str}\n</tool_response>"

    def is_tool_call(self, content: str) -> bool:
        if "<tool_call>" in content and "</tool_call>" in content and content.index("<tool_call>") < content.index("</tool_call>"):
            return True
        return False

    def is_terminate(self, content: str) -> bool:
        if "<answer>" in content and "</answer>" in content and content.index("<answer>") < content.index("</answer>"):
            return True

        if self.cur_step >= self.max_steps:
            return True

        return False

    def react(self) -> bool:
        self.cur_step += 1

        # think
        content = self.think()
        if not content:
            return False

        if self.model.startswith("qwen3") and "thinking" in self.model:
            if not content.startswith("<think>"):
                content = "<think>" + content
            if content.startswith("</think>"):
                content = content[len("</think>"):]

        # trace
        trace = {
            "data_source": "shopping_companion",
            "prompt": self.messages,
            "ability": "agent",
            "reward_model": {
                "style": "rule",
                "ground_truth": content,
            },
            "extra_info": {
                "max_steps": self.max_steps,
                "cur_steps": self.cur_step,
                "question_id": self.conversation.question_id,
                "question_type": self.conversation.question_type,
                "question": self.conversation.question,
                "question_date": self.conversation.question_date,
                "timestamp": int(time.time() * 1000),
                "model": self.model,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
        }
        self.tracker.append(copy.deepcopy(trace))

        # validate
        self.add_message("assistant", content)
        if content.count("<tool_call>") != content.count("</tool_call>"):
            self.add_message("user", "Please include your tool calls within <tool_call></tool_call> tags.")
            return True
        elif content.count("<answer>") != content.count("</answer>"):
            self.add_message("user", "Please include your final answer within <answer></answer> tags.")
            return True
        elif "<answer>" not in content and "<tool_call>" not in content:
            self.add_message("user", "In each turn you can either:\n- Think and make one or more tool calls.\n- Provide your final answer and terminate the conversation.\nYou cannot do both at the same time.")
            return True

        # act
        if self.is_tool_call(content):
            tool_response = self.act(content)
            self.add_message("user", tool_response)

        # terminate
        if self.is_terminate(content):
            return False

        return True

    def run(self):
        raise NotImplementedError


def producer(queue: Queue, hyp_file: str, ref_file: str, threads: int):
    # Load had questions from hypothesis file
    had_question_ids = set()
    if os.path.exists(hyp_file):
        with open(hyp_file, "r") as fin:
            portalocker.lock(fin, portalocker.LOCK_EX)
            for line in fin:
                jsonarr = json.loads(line.strip())
                if not jsonarr:
                    continue
                question_id = jsonarr[0].get("extra_info", {}).get("question_id")
                if not question_id:
                    continue
                had_question_ids.add(question_id)
            portalocker.unlock(fin)

    # Load remaining questions from reference file
    with open(ref_file, "r") as fin:
        for line in tqdm(fin, desc="Loading remaining questions from reference file: "):
            item = json.loads(line.strip())
            item = ConversationDTO(**item)

            if item.question_id in had_question_ids:
                continue

            queue.put(item)
            had_question_ids.add(item.question_id)

    # Put None to the queue to indicate the end of the questions
    for _ in range(threads):
        queue.put(None)


def consumer(
    AgentClass: AgentLoop,
    queue: Queue,
    hyp_file: str,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    while True:
        item = queue.get()
        if item is None:
            break
        conversation_id = item.question_id.rsplit("_", 1)[0]
        agent = AgentClass(
            conversation_id=conversation_id,
            conversation=item,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        agent.run()
        with open(hyp_file, "a") as fout:
            portalocker.lock(fout, portalocker.LOCK_EX)
            fout.write(json.dumps(agent.tracker) + "\n")
            portalocker.unlock(fout)
