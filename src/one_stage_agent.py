import argparse
import threading
from queue import Queue

from agent.agent_loop import AgentLoop, producer, consumer
from tools import all_tools
from prompt.agent import one_stage_prompt


class OneStageAgent(AgentLoop):
    @property
    def one_stage_prompt(self) -> str:
        tools = [toolclass() for toolclass in all_tools]
        available_tools = [tool for tool in tools]
        available_tools_str = "\n\n".join([tool.to_string() for tool in available_tools])
        return one_stage_prompt.format(available_tools=available_tools_str).strip()

    def run(self):
        # question
        query = f"Current Date: {self.conversation.question_date}\n{self.conversation.question}"

        self.add_message("system", self.one_stage_prompt)
        self.add_message("user", query)
        while self.cur_step <= self.max_steps:
            should_continue = self.react()
            if not should_continue:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_file", type=str, required=True)
    parser.add_argument("--ref_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    queue = Queue(args.threads)

    # Create processes
    producer_process = threading.Thread(target=producer, args=(queue, args.hyp_file, args.ref_file, args.threads))
    consumers = []
    for _ in range(args.threads):
        consumers.append(threading.Thread(target=consumer, args=(OneStageAgent, queue, args.hyp_file, args.model)))

    # Start processes
    producer_process.start()
    for consumer_process in consumers:
        consumer_process.start()

    # Join processes
    producer_process.join()
    for consumer_process in consumers:
        consumer_process.join()
