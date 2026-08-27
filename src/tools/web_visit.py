import os
import sys
import time
import asyncio
from typing import List

import tiktoken
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode

from tools.base import BaseTool
from prompt.summary import web_visit_summary_prompt, response_format
from util.llm import ParallelOpenAICompletion, CompletionRequest, parse_json_response


MAX_RETRIES = int(os.getenv("WEB_VISIT_MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("WEB_VISIT_RETRY_DELAY", 1.0))
MAX_TOKENS = int(os.getenv("WEB_VISIT_MAX_TOKENS", 95000))
MAX_WORKERS = int(os.getenv("WEB_VISIT_MAX_WORKERS", 10))
MODEL = os.getenv("WEB_VISIT_MODEL", "gpt-5-mini-2025-08-07-GlobalStandard")
TIMEOUT = int(os.getenv("WEB_VISIT_TIMEOUT", 60))
DEBUG = int(os.getenv("WEB_VISIT_DEBUG", 0))


browser_config = BrowserConfig()
run_config = CrawlerRunConfig(
    # Content filtering
    word_count_threshold=10,
    excluded_tags=['form', 'header'],
    exclude_external_links=True,

    # Content processing
    process_iframes=True,
    remove_overlay_elements=True,

    # Cache control
    cache_mode=CacheMode.ENABLED  # Use cache if available
)


async def batch_fetch_webpage_content(urls: List[str]) -> List[str]:
    if not urls:
        return []

    async def _fetch_single(target_url: str) -> str:
        for _ in range(MAX_RETRIES):
            try:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(url=target_url, config=run_config)
                    if result.success and result.markdown:
                        return result.markdown
                    return ""
            except Exception as e:
                print(f"Error occurred during fetching webpage content: {e}", file=sys.stderr)
                time.sleep(RETRY_DELAY)
        return ""

    tasks = [_fetch_single(target_url) for target_url in urls]
    results = await asyncio.gather(*tasks)
    return results


def truncate_webpage_content(content: str) -> str:
    encoding = tiktoken.get_encoding("o200k_base")

    tokens = encoding.encode(content)
    if len(tokens) <= MAX_TOKENS:
        return content

    truncated_tokens = tokens[:MAX_TOKENS]
    return encoding.decode(truncated_tokens)


def batch_summarize_webpage_content(urls: List[str], contents: List[str], goal: str) -> str:
    client = ParallelOpenAICompletion(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_workers=MAX_WORKERS,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY,
        timeout=TIMEOUT,
    )

    indices = []
    llm_requests = []
    for i, content in enumerate(contents):
        if len(content) > 0:
            prompt = web_visit_summary_prompt.format(webpage_content=content, goal=goal)
            if DEBUG:
                print(f"Prompt: {prompt}", file=sys.stderr)
            indices.append(i)
            llm_requests.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model=MODEL,
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                    response_format=response_format,
                )
            )
    llm_responses = client.batch_complete(llm_requests)

    idx2summary = {}
    for i, response in zip(indices, llm_responses):
        if response.success:
            jsonobj = parse_json_response(response.content)
            if jsonobj and jsonobj.get("evidence") and jsonobj.get("summary"):
                idx2summary[i] = {
                    "evidence": jsonobj.get("evidence"),
                    "summary": jsonobj.get("summary"),
                }
        else:
            print(f"Error summarizing webpage content from: {response.error}")

    template = 'The useful information in "{url}" for goal "{goal}" is as follows:\n\n'
    template += "**Evidence in webpage**: \n{evidence}\n\n"
    template += "**Summary**: \n{summary}"
    evidence = "The provided webpage content could not be accessed. Please check the URL or file format."
    summary = "The webpage content could not be processed, and therefore, no information is available."
    tool_responses = []
    for i, url in enumerate(urls):
        if i in idx2summary:
            tool_responses.append(template.format(url=url, goal=goal, evidence=idx2summary[i]["evidence"], summary=idx2summary[i]["summary"]))
        else:
            tool_responses.append(template.format(url=url, goal=goal, evidence=evidence, summary=summary))
    delimiter = "\n\n" + "=" * 10 + "\n\n"
    return delimiter.join(tool_responses).strip()


class WebVisit(BaseTool):
    name = "web_visit"
    description = "Visit webpage(s) and return the summary of the content."
    parameters = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string", "description": "The webpage URL."},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit.",
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s).",
            },
        },
        "required": ["urls", "goal"],
    }

    def execute(self, **kwargs) -> str:
        urls = kwargs.get("urls")
        goal = kwargs.get("goal")

        if not urls:
            return "No URLs provided."
        if not isinstance(urls, list):
            return "URLs must be a list of strings."
        if not goal:
            return "No goal provided."

        contents = asyncio.run(batch_fetch_webpage_content(urls))
        contents = [truncate_webpage_content(content) for content in contents]
        tool_responses = batch_summarize_webpage_content(urls, contents, goal)
        return tool_responses
