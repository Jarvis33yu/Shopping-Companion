import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from tools.base import BaseTool


AI_HUB_SEARCH_BASE_URL = os.getenv("AI_HUB_SEARCH_BASE_URL")
AI_HUB_SEARCH_TOKEN = os.getenv("AI_HUB_SEARCH_TOKEN")
MAX_RETRIES = int(os.getenv("WEB_SEARCH_MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("WEB_SEARCH_RETRY_DELAY", 1.0))
TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", 60))
MAX_WORKERS = int(os.getenv("WEB_SEARCH_MAX_WORKERS", 10))


def parse_google_search_page(idx: int, page: dict) -> str:
    title = page.get("title", "")
    link = page.get("link", "")
    snippet = page.get("snippet", "")
    source = page.get("source", "")
    date_published = page.get("date", "")

    date_str = ""
    if date_published:
        date_str = f"\nDate published: {date_published}"

    source_str = ""
    if source:
        source_str = f"\nSource: {source}"

    snippet_str = ""
    if snippet:
        snippet_str = f"\n{snippet}"

    redacted_version = (
        f"{idx}. [{title}]({link}){date_str}{source_str}{snippet_str}"
    )
    redacted_version = redacted_version.replace(
        "Your browser can't play this video.", ""
    )
    return redacted_version.strip()


def google_search(query: str) -> str:
    url = f"{AI_HUB_SEARCH_BASE_URL}/customsearch/google/search"
    headers = {
        "Authorization": f"Bearer {AI_HUB_SEARCH_TOKEN}",
        "Content-Type": "application/json",
    }
    body = {"q": query}

    for i in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
            response.raise_for_status()
            results = response.json()

            idx = 0
            web_snippets = []
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    web_snippets.append(parse_google_search_page(idx, page))
            else:
                for value in results.values():
                    if isinstance(value, list) and len(value) > 0:
                        for page in value:
                            if isinstance(page, dict):
                                idx += 1
                                web_snippets.append(parse_google_search_page(idx, page))
            if web_snippets:
                tool_response = f'A web search for "{query}" found {len(web_snippets)} results:\n\n' + "\n\n".join(web_snippets)
                break
            else:
                tool_response = f'Web search for "{query}" found no results.'
        except Exception as e:
            print(f"Error occurred during web search: {e}, retry {i + 1}/{MAX_RETRIES}", file=sys.stderr)
            tool_response = f"Error occurred during web search: {e}"
            time.sleep(RETRY_DELAY)

    return tool_response


class WebSearch(BaseTool):
    name = "web_search"
    description = 'Performs batched web searches: supply an array of query; the tool retrieves the top 10 results for each query in one call.'
    parameters = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "An array of query strings.",
            },
        },
        "required": ["queries"],
    }

    def execute(self, **kwargs) -> str:
        queries = kwargs.get("queries")

        if not queries:
            return "No queries provided."
        if not isinstance(queries, list):
            return "Queries must be a list of strings."

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tool_responses = list(executor.map(google_search, queries))

        delimiter = "\n\n" + "=" * 10 + "\n\n"
        return delimiter.join(tool_responses).strip()
