import argparse
import threading
from queue import Queue

from agent.agent_loop import producer, consumer
from two_stage_agent import TwoStageAgent


class OracleAgent(TwoStageAgent):
    def run(self):
        # question
        query = f"Current Date: {self.conversation.question_date}\n{self.conversation.question}"

        # oracle
        oracle = []
        for answer_session_id in self.conversation.answer_session_ids:
            for session_id, session, date in zip(
                self.conversation.haystack_session_ids,
                self.conversation.haystack_sessions,
                self.conversation.haystack_dates,
            ):
                if session_id == answer_session_id:
                    session_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in session])
                    oracle.append(f"[Date: {date}]\n{session_str}")
        oracle_str = "The most relevant user dialogue memories are as follows:\n\n" + "\n\n".join(oracle)

        self.add_message("system", self.two_stage_2_prompt)
        self.add_message("user", query)
        self.add_message("user", oracle_str)
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
        consumers.append(threading.Thread(target=consumer, args=(OracleAgent, queue, args.hyp_file, args.model)))

    # Start processes
    producer_process.start()
    for consumer_process in consumers:
        consumer_process.start()

    # Join processes
    producer_process.join()
    for consumer_process in consumers:
        consumer_process.join()
