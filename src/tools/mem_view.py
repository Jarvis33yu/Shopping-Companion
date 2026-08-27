import os
import sys
import time
from typing import List, Tuple

import requests

from tools.base import BaseTool


MEM_VIEW_BASE_URL = os.getenv("MEM_VIEW_BASE_URL", "http://127.0.0.1:5632")
TIMEOUT = int(os.getenv("MEM_VIEW_TIMEOUT", 60))
MAX_RETRIES = int(os.getenv("MEM_VIEW_MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("MEM_VIEW_RETRY_DELAY", 1.0))


def fetch_single_session(index: int, conversation_id: str) -> Tuple[List[int], str]:
    for i in range(MAX_RETRIES):
        try:
            url = f"{MEM_VIEW_BASE_URL}/view_session?conversation_id={conversation_id}&idx={index}"
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            indices, results = resp.json()
            return indices, results
        except Exception as e:
            print(f'View session for memory index "{index}" failed: {e}, retry {i + 1}/{MAX_RETRIES}', file=sys.stderr)
            time.sleep(RETRY_DELAY)
    return [], f'View session for memory index "{index}" failed.'


class MemView(BaseTool):
    name: str = "mem_view"
    description: str = "Given a array of memory indices, retrieve the dialogue session containing those memories."
    parameters: dict = {
        "type": "object",
        "properties": {
            "indices": {
                "type": "array",
                "items": {"type": "integer", "description": "The index of the memory."},
                "minItems": 1,
                "description": "An array of memory indices.",
            },
        },
        "required": ["indices"],
    }

    def execute(self, **kwargs):
        indices = kwargs.get("indices")
        conversation_id = kwargs.get("conversation_id")

        if not indices:
            return "No indices provided."
        if not isinstance(indices, list):
            return "Indices must be a array of integers."

        tool_responses = []

        ind2sess = {}
        for index in indices:
            sess_indices, sess = fetch_single_session(index, conversation_id)

            if not sess_indices:
                tool_responses.append(sess)
                continue

            key = tuple(sess_indices)
            if key not in ind2sess:
                ind2sess[key] = sess

        for key, sess in ind2sess.items():
            indices_str = ", ".join([str(i) for i in key])
            tool_responses.append(f'The dialogue session includes memory indices "{indices_str}" is:\n\n{sess}')

        delimiter = "\n\n" + "=" * 10 + "\n\n"
        return delimiter.join(tool_responses).strip()
