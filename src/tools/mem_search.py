import os
import sys
import time
from urllib.parse import quote_plus

import requests

from tools.base import BaseTool


MEM_SEARCH_BASE_URL = os.getenv("MEM_SEARCH_BASE_URL", "http://127.0.0.1:5632")
TIMEOUT = int(os.getenv("MEM_SEARCH_TIMEOUT", 60))
MAX_RETRIES = int(os.getenv("MEM_SEARCH_MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("MEM_SEARCH_RETRY_DELAY", 1.0))


def single_mem_search(query: str, conversation_id: str) -> str:
    for i in range(MAX_RETRIES):
        try:
            url = f"{MEM_SEARCH_BASE_URL}/search?conversation_id={conversation_id}&query={quote_plus(query)}&k=10"
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            indices, result = resp.json()

            if indices and result:
                return f'A memory search for "{query}" found {len(indices)} results:\n\n' + result

            return f'Memory search for "{query}" found no results.'
        except Exception as e:
            print(f"Single memory search error: {e}, retry {i + 1}/{MAX_RETRIES}", file=sys.stderr)
            time.sleep(RETRY_DELAY)

    return f'Memory Search for "{query}" timed out. Please try again later.'


class MemSearch(BaseTool):
    name: str = "mem_search"
    description: str = 'Batch search for dialogue memories (conversation turns) based on a array of queries; the tool retrieves the top 10 results for each query in one call.'
    parameters: dict = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "Array of query strings.",
            },
        },
        "required": ["queries"],
    }

    def execute(self, **kwargs):
        queries = kwargs.get("queries")
        conversation_id = kwargs.get("conversation_id")

        if not queries:
            return "No queries provided."
        if not isinstance(queries, list):
            return "Queries must be a list of strings."

        results = []
        for q in queries:
            result = single_mem_search(q, conversation_id)
            results.append(result)

        delimiter = "\n\n" + "=" * 10 + "\n\n"
        return delimiter.join(results).strip()
