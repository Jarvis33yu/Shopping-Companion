import os
import sys
import time
from datetime import datetime

import requests

from tools.base import BaseTool
from util.misc import convert_date_to_timestamp
from prompt.summary import mem_summary_prompt, response_format
from util.llm import ParallelOpenAICompletion, CompletionRequest, parse_json_response


MEM_SUMMARIZE_BY_DATE_BASE_URL = os.getenv("MEM_SUMMARIZE_BY_DATE_BASE_URL", "http://127.0.0.1:5632")
TIMEOUT = int(os.getenv("MEM_SUMMARIZE_BY_DATE_TIMEOUT", 60))
MAX_RETRIES = int(os.getenv("MEM_SUMMARIZE_BY_DATE_MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("MEM_SUMMARIZE_BY_DATE_RETRY_DELAY", 1.0))
MAX_WORKERS = int(os.getenv("MEM_SUMMARIZE_BY_DATE_MAX_WORKERS", 10))
MODEL = os.getenv("MEM_SUMMARIZE_BY_DATE_MODEL", "gpt-5-mini-2025-08-07-GlobalStandard")
DEBUG = int(os.getenv("MEM_SUMMARIZE_BY_DATE_DEBUG", 0))


class MemSummarizeByDate(BaseTool):
    name: str = "mem_summarize_by_date"
    description: str = "Given a %Y-%m-%d formatted start date, summarize sessions within the following offset days according to the specified goal."
    parameters: dict = {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "The beginning date of the range.",
            },
            "offset": {
                "type": "integer",
                "description": "The number of days to include after the start date.",
                "minimum": 1,
                "maximum": 7,
            },
            "goal": {
                "type": "string",
                "description": "The goal of the summary.",
            },
        },
        "required": ["start_date", "offset", "goal"],
    }

    def execute(self, **kwargs):
        start_date = kwargs.get("start_date")
        offset = kwargs.get("offset")
        goal = kwargs.get("goal")
        conversation_id = kwargs.get("conversation_id")

        if not start_date:
            return "No start date provided."
        if not offset:
            return "No offset provided."
        if not isinstance(offset, int) or (isinstance(offset, str) and not offset.isdigit()):
            return "Offset must be an integer."
        offset = int(offset)
        if offset < 1 or offset > 7:
            return "Offset must be between 1 and 7."
        if not goal:
            return "No goal provided."

        client = ParallelOpenAICompletion(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=MAX_WORKERS,
            max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY,
        )

        for i in range(MAX_RETRIES):
            try:
                start_timestamp = convert_date_to_timestamp(start_date)
                end_timestamp = start_timestamp + offset * 24 * 60 * 60
                end_date = datetime.fromtimestamp(end_timestamp).strftime("%Y-%m-%d")
                url = f"{MEM_SUMMARIZE_BY_DATE_BASE_URL}/view_sessions_by_date?conversation_id={conversation_id}&start_date={start_date}&end_date={end_date}"
                resp = requests.get(url, timeout=TIMEOUT)
                resp.raise_for_status()
                sess_indices, sess_results = resp.json()

                assert len(sess_results) > 0 and len(sess_results) == len(sess_indices), "No sessions found."

                dates = []
                llm_requests = []
                for session in sess_results:
                    dates.append(session.split("\n")[0].strip("[]"))
                    prompt = mem_summary_prompt.format(conversation=session, goal=goal)
                    if DEBUG:
                        print(f"Prompt: {prompt}", file=sys.stderr)
                    llm_requests.append(
                        CompletionRequest(
                            messages=[{"role": "user", "content": prompt}],
                            model=MODEL,
                            extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                            response_format=response_format,
                        )
                    )
                tool_responses = []
                results = client.batch_complete(llm_requests, preserve_order=True)
                for result, date in zip(results, dates):
                    jsonobj = parse_json_response(result.content)
                    if not result.success or not jsonobj or not jsonobj.get("evidence") or not jsonobj.get("summary"):
                        tool_responses.append(f"Failed to summarize the session on {date}: {result.error}")
                    else:
                        evidence = jsonobj.get("evidence", "")
                        summary = jsonobj.get("summary", "")
                        tool_responses.append(f'The useful information for goal "{goal}" on {date} is as follows:\n\n**Evidence in session**:\n{evidence}\n\n**Summary**:\n{summary}')
                break
            except Exception as e:
                print(f"Summarize sessions for the range starting '{start_date}' and ending '{end_date}' failed: {e}, retry {i + 1}/{MAX_RETRIES}", file=sys.stderr)
                tool_responses = [f"Summarize sessions for the range starting '{start_date}' and ending '{end_date}' failed: {e}"]
                time.sleep(RETRY_DELAY)
        delimiter = "\n\n" + "=" * 10 + "\n\n"
        return delimiter.join(tool_responses).strip()
