import argparse
import threading
from queue import Queue

from agent.agent_loop import AgentLoop, producer, consumer
from tools import all_tools
from prompt.agent import two_stage_1_prompt, two_stage_2_prompt, qwen3_tool_template


class TwoStageAgent(AgentLoop):
    @property
    def two_stage_1_prompt(self) -> str:
        tools = [toolclass() for toolclass in all_tools]
        available_tools = [tool for tool in tools if tool.name in {"mem_search", "mem_view", "mem_summarize_by_date"}]

        if self.model.startswith("qwen3"):
            available_tools_str = qwen3_tool_template.format(available_tools="\n".join([tool.to_qwen3_string() for tool in available_tools]))
        else:
            available_tools_str = "# Tools\n\n" + "\n\n".join([tool.to_string() for tool in available_tools])

        return two_stage_1_prompt.format(available_tools=available_tools_str).strip()

    @property
    def two_stage_2_prompt(self) -> str:
        tools = [toolclass() for toolclass in all_tools]
        available_tools = [tool for tool in tools if tool.name in {"web_search", "web_visit", "product_search", "product_view"}]

        if self.model.startswith("qwen3"):
            available_tools_str = qwen3_tool_template.format(available_tools="\n".join([tool.to_qwen3_string() for tool in available_tools]))
        else:
            available_tools_str = "# Tools\n\n" + "\n\n".join([tool.to_string() for tool in available_tools])

        return two_stage_2_prompt.format(available_tools=available_tools_str).strip()

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

        self.clear_messages()
        self.add_message("system", self.two_stage_2_prompt)
        self.add_message("user", query)
        self.add_message("assistant", answer_content)
        self.add_message("user", "I will not provide you with any more information. Please complete the task.")
        while self.cur_step <= self.max_steps:
            should_continue = self.react()
            if not should_continue:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_file", type=str, required=True)
    parser.add_argument("--ref_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    queue = Queue(args.threads)

    # Create processes
    producer_process = threading.Thread(target=producer, args=(queue, args.hyp_file, args.ref_file, args.threads))
    consumers = []
    for _ in range(args.threads):
        consumers.append(threading.Thread(target=consumer, args=(TwoStageAgent, queue, args.hyp_file, args.model, args.base_url, args.api_key)))

    # Start processes
    producer_process.start()
    for consumer_process in consumers:
        consumer_process.start()

    # Join processes
    producer_process.join()
    for consumer_process in consumers:
        consumer_process.join()
