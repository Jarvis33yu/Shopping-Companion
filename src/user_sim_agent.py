import os
import sys
import argparse
import threading
from queue import Queue
from typing import Any

from agent.agent_loop import producer, consumer
from two_stage_agent import TwoStageAgent
from tools import all_tools
from prompt.agent import two_stage_2_prompt
from prompt.user_sim import missing_wrong_user_prompt, feature_names_user_prompt
from util.llm import ParallelOpenAICompletion, CompletionRequest, parse_json_response


args: Any = None

client: Any = ParallelOpenAICompletion(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    max_workers=1,
    max_retries=3,
    retry_delay=1,
)


def hint_missing_or_wrong(hypothesis: str, reference: str) -> str:
    prompt = missing_wrong_user_prompt.format(hypothesis=hypothesis, reference=reference)
    llm_request = CompletionRequest(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-5-2025-08-07-GlobalStandard",
        extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
    )
    result = client.batch_complete([llm_request])[0]
    if result.success:
        hint = result.content.strip()

        if hint == "all matched":
            return "I will not provide you with any more information. Please complete the task."
        elif hint == "missing":
            return "Some preferences are missing. Re-check the memories and find the missing preferences, then complete the task. Do not ask me to provide more information."
        elif hint == "wrong":
            return "Some preferences are wrong. Re-check the memories and find the wrong preferences, then complete the task. Do not ask me to provide more information."
        else:
            raise ValueError(f"Invalid hint: {hint}")
    else:
        print(result.error, file=sys.stderr)
        return ""


def hint_feature_names(hypothesis: str, reference: str) -> str:
    prompt = feature_names_user_prompt.format(hypothesis=hypothesis, reference=reference)
    llm_request = CompletionRequest(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-5-2025-08-07-GlobalStandard",
        extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
    )
    result = client.batch_complete([llm_request])[0]
    if result.success:
        jsonobj = parse_json_response(result.content)

        missing = jsonobj.get("missing", [])
        wrong = jsonobj.get("wrong", [])

        hint = ""
        if missing:
            hint += f"These preferences are missing: {', '.join(missing)}. \n"
        if wrong:
            hint += f"These preferences are wrong: {', '.join(wrong)}. \n"

        if not hint:
            hint = "I will not provide you with any more information. Please complete the task."
        else:
            hint = f"{hint}\Re-check the memories and find the missing or wrong preferences, then complete the task. Do not ask me to provide more information."

        return hint

    else:
        print(result.error, file=sys.stderr)
        return ""


class UserSimAgent(TwoStageAgent):
    @property
    def two_stage_2_prompt(self) -> str:
        tools = [toolclass() for toolclass in all_tools]
        available_tools = [tool for tool in tools]
        available_tools_str = "\n\n".join([tool.to_string() for tool in available_tools])
        return two_stage_2_prompt.format(available_tools=available_tools_str).strip()

    @property
    def reference_content(self) -> str:
        reference_content = ""
        if self.conversation.question_type == "single_product":
            product_name = self.conversation.answer["product_name"]
            wanted_features = self.conversation.answer["wanted_features"]
            wanted_features_str = ""
            for feature in wanted_features:
                wanted_features_str += f"- {feature}\n"
            reference_content = f"Product Name: {product_name}\nWanted Features:\n{wanted_features_str}\n"
        elif self.conversation.question_type == "add_on_deals":
            for i, preference in enumerate(self.conversation.answer["preferences"]):
                product_name = preference["product_name"]
                wanted_features = preference["wanted_features"]
                wanted_features_str = ""
                for feature in wanted_features:
                    wanted_features_str += f"- {feature}\n"
                reference_content += f"{i+1}. Product Name: {product_name}\nWanted Features:\n{wanted_features_str}\n"
        else:
            raise ValueError(f"Unsupported question type: {self.conversation.question_type}")

        return reference_content

    def run(self):
        # question
        query = f"Current Date: {self.conversation.question_date}\n{self.conversation.question}"

        # stage 1: retrieval
        self.add_message("system", self.two_stage_1_prompt)
        self.add_message("user", query)
        while self.cur_step <= self.max_steps:
            should_continue = self.react()
            if not should_continue:
                break

        # stage 2: complete task
        if not self.messages:
            return

        if self.cur_step >= self.max_steps:
            return

        answer_content = self.messages[-1]["content"]
        if not self.is_terminate(answer_content):
            return
        answer_content = answer_content.split("<answer>")[1].split("</answer>")[0].strip()
        hint_content = ""
        while not hint_content:
            if args.user_mode == "missing_or_wrong":
                hint_content = hint_missing_or_wrong(answer_content, self.reference_content)
            elif args.user_mode == "feature_names":
                hint_content = hint_feature_names(answer_content, self.reference_content)
            else:
                raise ValueError(f"Invalid user mode: {args.user_mode}")
        self.clear_messages()
        self.add_message("system", self.two_stage_2_prompt)
        self.add_message("user", query)
        self.add_message("assistant", answer_content)
        self.add_message("user", hint_content)
        while self.cur_step <= self.max_steps:
            should_continue = self.react()
            if not should_continue:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_file", type=str, required=True)
    parser.add_argument("--ref_file", type=str, required=True)
    parser.add_argument("--user_mode", type=str, required=True, choices=["missing_or_wrong", "feature_names"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    queue = Queue(args.threads)

    # Create processes
    producer_process = threading.Thread(target=producer, args=(queue, args.hyp_file, args.ref_file, args.threads))
    consumers = []
    for _ in range(args.threads):
        consumers.append(threading.Thread(target=consumer, args=(UserSimAgent, queue, args.hyp_file, args.model)))

    # Start processes
    producer_process.start()
    for consumer_process in consumers:
        consumer_process.start()

    # Join processes
    producer_process.join()
    for consumer_process in consumers:
        consumer_process.join()
