import json
import logging
import os
import sys
from typing import Any, Optional, Tuple, Dict
from uuid import uuid4
import time
import requests
from urllib.parse import quote_plus
from verl.utils.rollout_trace import rollout_trace_op
import tiktoken
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from prompt.summary import mem_summary_prompt, response_format, web_visit_summary_prompt
from util.llm import CompletionRequest, ParallelOpenAICompletion, parse_json_response
from util.misc import convert_date_to_timestamp
import httpx
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode

import asyncio
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

REWARD_BASE_URL = os.getenv("REWARD_BASE_URL", "http://127.0.0.1:5633")
MEM_BASE_URL = os.getenv("MEM_BASE_URL", "http://127.0.0.1:5632")
PRODUCT_BASE_URL = os.getenv("PRODUCT_BASE_URL", "http://127.0.0.1:5631")
AI_HUB_SEARCH_BASE_URL = os.getenv("AI_HUB_SEARCH_BASE_URL")
AI_HUB_SEARCH_TOKEN = os.getenv("AI_HUB_SEARCH_TOKEN")
TIMEOUT = int(os.getenv("MEM_SEARCH_TIMEOUT", 60))
MAX_WORKERS = int(os.getenv("MEM_SUMMARIZE_BY_DATE_MAX_WORKERS", 10))
MAX_TOKENS = int(os.getenv("WEB_VISIT_MAX_TOKENS", 95000))
MAX_RETRIES = int(os.getenv("MEM_SEARCH_MAX_RETRIES", 3))
RETRY_DELAY = float(os.getenv("MEM_SEARCH_RETRY_DELAY", 1.0))
MODEL = os.getenv("MEM_SUMMARIZE_BY_DATE_MODEL", "gpt-5-mini-2025-08-07-GlobalStandard")
DEBUG = int(os.getenv("MEM_SUMMARIZE_BY_DATE_DEBUG", 0))

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

# 全局共享客户端字典
_SHARED_CLIENTS: Dict[str, httpx.AsyncClient] = {}

def get_shared_client(base_url: str, timeout: float = 60.0) -> httpx.AsyncClient:

    if base_url not in _SHARED_CLIENTS:
        _SHARED_CLIENTS[base_url] = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=10
            ),
            transport=httpx.AsyncHTTPTransport(retries=3)
        )
    return _SHARED_CLIENTS[base_url]

async def close_all_clients():

    for client in _SHARED_CLIENTS.values():
        await client.aclose()
    _SHARED_CLIENTS.clear()

class MemSearch(BaseTool):
    """Search for the k most similar memories based on the query."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._client = get_shared_client(MEM_BASE_URL, timeout=TIMEOUT)
        self.reward_client = get_shared_client(REWARD_BASE_URL, timeout=TIMEOUT)
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", kwargs)
        conversation_id = create_kwargs.get("conversation_id")
        question_id = create_kwargs.get("question_id")
        self._instance_dict[instance_id] = {
            "response": "",
            "conversation_id": conversation_id,
            "question_id": question_id,
            }
        return instance_id, ToolResponse()
    
    async def single_mem_search(self, query: str, conversation_id: str) -> str:
        for i in range(MAX_RETRIES):
            try:
                url = f"/search?conversation_id={conversation_id}&query={quote_plus(query)}&k=10"
                resp = await self._client.get(url)
                resp.raise_for_status()
                indices, result = resp.json()
                # async with httpx.AsyncClient() as client:
                #     resp = await client.get(url, timeout=TIMEOUT)
                #     resp.raise_for_status()
                #     indices, result = resp.json()

                if indices and result:
                    return f'A memory search for "{query}" found {len(indices)} results:\n\n' + result

                return f'Memory search for "{query}" found no results.'
            except Exception as e:
                print(f"Single memory search error: {e}, retry {i + 1}/{MAX_RETRIES}", file=sys.stderr)
                await asyncio.sleep(RETRY_DELAY) 

        return f'Memory Search for "{query}" timed out. Please try again later.'

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:

        conversation_id = self._instance_dict[instance_id].get("conversation_id")

        self._instance_dict[instance_id]["parameters"] = parameters

        queries = parameters.get("queries")

        if not queries:
            error_msg = "Error: 'queries' is required."
            logger.error(f"[mem_search] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}

        if not conversation_id:
            error_msg = "Error: 'conversation_id' is not set."
            logger.error(f"[mem_search] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}

        tasks = [self.single_mem_search(q, conversation_id) for q in queries]
        results = await asyncio.gather(*tasks)

        delimiter = "\n\n" + "-" * 10 + "\n\n"
        result_text = delimiter.join(results).strip()

        self._instance_dict[instance_id]["response"] = result_text

        reward = await self.calc_reward(instance_id)

        return ToolResponse(text=result_text), reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        try:
            entry = self._instance_dict.get(instance_id)
            if not entry:
                return 0.0
            question_id = entry.get("question_id")
            parameters = entry.get("parameters")
            if not question_id or not parameters:
                return 0.0

            queries = parameters.get("queries")
            if not queries:
                return 0.0

            payload = {
                "question_id": question_id,
                "name": "mem_search",
                "kwargs": {"queries": queries}
            }

            resp = await self.reward_client.post("/reward", json=payload)
            if resp.status_code != 200:
                logger.warning(f"[MemSearch] Reward server returned {resp.status_code}: {resp.text}")
                return 0.0

            data = resp.json()
            return float(data.get("reward", 0.0))

        except Exception as e:
            logger.error(f"[MemSearch] Failed to calculate reward: {e}")
            return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class MemView(BaseTool):
    """Tool for viewing memories around a specific index."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._client = get_shared_client(MEM_BASE_URL, timeout=TIMEOUT)
        self.reward_client = get_shared_client(REWARD_BASE_URL, timeout=TIMEOUT)
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", kwargs)
        conversation_id = create_kwargs.get("conversation_id")
        question_id = create_kwargs.get("question_id")
        self._instance_dict[instance_id] = {
            "response": "",
            "conversation_id": conversation_id,
            "question_id": question_id,
            }
        return instance_id, ToolResponse()
    
    async def fetch_single_session(self, index: int, conversation_id: str) -> Tuple[List[int], str]:
        for i in range(MAX_RETRIES):
            try:
                url = f"/view_session?conversation_id={conversation_id}&idx={index}"
                # async with httpx.AsyncClient() as client:
                    # resp = await client.get(url, timeout=TIMEOUT)
                    # resp.raise_for_status()
                    # indices, results = resp.json()
                resp = await self._client.get(url)
                resp.raise_for_status()
                indices, results = resp.json()
                return indices, results
            except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
                logger.error(f'View session for memory index "{index}" failed: {e}, retry {i + 1}/{MAX_RETRIES}')
                if i < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)  
        return [], f'View session for memory index "{index}" failed.'

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        conversation_id = self._instance_dict[instance_id].get("conversation_id")
        indices = parameters.get("indices")
        self._instance_dict[instance_id]["parameters"] = parameters

        if indices is None:
            error_msg = "Error: 'indices' is required."
            logger.error(f"[mem_view] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}

        if not conversation_id:
            error_msg = "Error: 'conversation_id' is not set."
            logger.error(f"[mem_view] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}

        tool_responses = [] 
        
        ind2sess = {}
        for index in indices:
            sess_indices, sess = await self.fetch_single_session(index, conversation_id)

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
        observation = delimiter.join(tool_responses).strip()
    

        self._instance_dict[instance_id]["response"] = observation

        reward = await self.calc_reward(instance_id)

        return ToolResponse(text=observation), reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        try:
            entry = self._instance_dict.get(instance_id)
            if not entry:
                return 0.0
            question_id = entry.get("question_id")
            parameters = entry.get("parameters")
            if not question_id or not parameters:
                return 0.0

            indices = parameters.get("indices")
            if not indices or not isinstance(indices, list):
                return 0.0
            # 过滤有效 index
            try:
                indices = [int(i) for i in indices if isinstance(i, (int, str)) and str(i).strip().isdigit()]
            except Exception:
                return 0.0
            if not indices:
                return 0.0

            payload = {
                "question_id": question_id,
                "name": "mem_view",
                "kwargs": {"indices": indices}
            }

            resp = await self.reward_client.post("/reward", json=payload)
            if resp.status_code != 200:
                logger.warning(f"[MemView] Reward server returned {resp.status_code}: {resp.text}")
                return 0.0

            data = resp.json()
            return float(data.get("reward", 0.0))

        except Exception as e:
            logger.error(f"[MemView] Failed to calculate reward: {e}")
            return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class MemSummarizeByDate(BaseTool):
    """Tool for viewing memories within a date range."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._client = get_shared_client(MEM_BASE_URL, timeout=TIMEOUT)
        self.reward_client = get_shared_client(REWARD_BASE_URL, timeout=TIMEOUT)
        self._instance_dict = {}
        
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", kwargs)
        conversation_id = create_kwargs.get("conversation_id")
        question_id = create_kwargs.get("question_id")
        self._instance_dict[instance_id] = {
            "response": "",
            "conversation_id": conversation_id,
            "question_id": question_id,
            }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        conversation_id = self._instance_dict[instance_id].get("conversation_id")
        start_date = parameters.get("start_date")
        offset = parameters.get("offset")
        goal = parameters.get("goal")
        self._instance_dict[instance_id]["parameters"] = parameters

        if not start_date:
            error_msg = "Error: 'start_date' is required."
            logger.error(f"[mem_summarize_by_date] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}
        
        if not offset:
            error_msg = "Error: 'offset' is required."
            logger.error(f"[mem_summarize_by_date] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}

        if not conversation_id:
            error_msg = "Error: 'conversation_id' is not set."
            logger.error(f"[mem_summarize_by_date] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}

        llm_client = ParallelOpenAICompletion(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=MAX_WORKERS,
            max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY,
        )

        for i in range(MAX_RETRIES):
            end_date = None
            try:
                start_timestamp = convert_date_to_timestamp(start_date)
                end_timestamp = start_timestamp + offset * 24 * 60 * 60
                end_date = datetime.fromtimestamp(end_timestamp).strftime("%Y-%m-%d")
                url = f"/view_sessions_by_date?conversation_id={conversation_id}&start_date={start_date}&end_date={end_date}"
                # async with httpx.AsyncClient() as client:
                #     resp = await client.get(url, timeout=TIMEOUT)
                #     resp.raise_for_status()
                #     sess_indices, sess_results = resp.json()
                resp = await self._client.get(url)
                resp.raise_for_status()
                sess_indices, sess_results = resp.json()


                assert len(sess_results) > 0 and len(sess_results) == len(sess_indices), "No sessions found."

                dates = []
                llm_requests = []
                for session in sess_results:
                    dates.append(session.split("\n")[0])
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
                results = await asyncio.to_thread(
                    llm_client.batch_complete, 
                    llm_requests, 
                    preserve_order=True
                )
                for result, date in zip(results, dates):
                    jsonobj = parse_json_response(result.content)
                    if not result.success or not jsonobj or not jsonobj.get("evidence") or not jsonobj.get("summary"):
                        tool_responses.append(f"Summarize the session for {date} failed: {result.error}")
                    else:
                        evidence = jsonobj.get("evidence", "")
                        summary = jsonobj.get("summary", "")
                        tool_responses.append(f"**Evidence in session**: \n{evidence}\n\n**Summary**: \n{summary}")
                break   
            except Exception as e:
                print(f"Summarize sessions for the range starting '{start_date}' and ending '{end_date}' failed: {e}, retry {i + 1}/{MAX_RETRIES}", file=sys.stderr)
                tool_responses = [f"Summarize sessions for the range starting '{start_date}' and ending '{end_date}' failed: {e}"]
                await asyncio.sleep(RETRY_DELAY) 
        delimiter = "\n\n" + "-" * 10 + "\n\n"
        observation = delimiter.join(tool_responses).strip()

        self._instance_dict[instance_id]["response"] = observation

        reward = await self.calc_reward(instance_id)

        return ToolResponse(text=observation), reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
     
        try:
            entry = self._instance_dict.get(instance_id)
            if not entry or not entry.get("question_id") or "parameters" not in entry:
                return 0.0
            question_id = entry["question_id"]
            params = entry["parameters"]

            start_date = params.get("start_date")
            offset = params.get("offset")
            goal = params.get("goal")

            if not start_date or not offset or not goal:
                return 0.0

            # 类型校验
            try:
                offset = int(offset)
                if offset < 1 or offset > 7:
                    return 0.0
            except Exception:
                return 0.0

            payload = {
                "question_id": question_id,
                "name": "mem_summarize_by_date",
                "kwargs": {
                    "start_date": start_date,
                    "offset": offset,
                    "goal": goal
                }
            }

            resp = await self.reward_client.post("/reward", json=payload)
            if resp.status_code != 200:
                logger.warning(f"[MemSummarizeByDate] Reward server returned {resp.status_code}: {resp.text}")
                return 0.0

            data = resp.json()
            return float(data.get("reward", 0.0))

        except Exception as e:
            logger.error(f"[MemSummarizeByDate] Failed to calculate reward: {e}")
            return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class ProductSearch(BaseTool):
    """Tool for finding memories similar to a specific memory."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._client = get_shared_client(PRODUCT_BASE_URL, timeout=TIMEOUT)
        self.reward_client = get_shared_client(REWARD_BASE_URL, timeout=TIMEOUT)
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", kwargs)
        question_id = create_kwargs.get("question_id")
        self._instance_dict[instance_id] = {
            "response": "",
            "question_id": question_id,
            }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        
        query = parameters.get("query")
        shop_id = parameters.get("shop_id")
        price = parameters.get("price")
        self._instance_dict[instance_id]["parameters"] = parameters

        if not query:
            error_msg = "Error: 'query' is required."
            logger.error(f"[product_search] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}
        
        shop_id_str = ""
        if shop_id:
            shop_id_str = f"&shop_id={shop_id}"
        
        price_str = ""
        if price:
            try:
                price_range = price.split("-")
                price_range = [float(p) if p else float('inf') for p in price_range]
                assert price_range[0] <= price_range[1]
                assert price_range[0] >= 0
            except Exception as e:
                print(f"Error occurred during parsing price range: {e}", file=sys.stderr)
                error_msg = "Invalid price range. Please provide a price range in the format of 'min-max' (e.g., '0-100') or 'min-' for no upper bound (e.g., '1000-')."
                return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}
            
            price_str = f"&price={price}"

        url = f"/product_search?query={quote_plus(query)}&k=50{shop_id_str}{price_str}"
        tool_response = "Error occurred during product search." 
        for i in range(MAX_RETRIES):
            try:
                # async with httpx.AsyncClient() as client:
                #     resp = await client.get(url, timeout=TIMEOUT)
                #     resp.raise_for_status()
                #     results = resp.json()
                resp = await self._client.get(url)
                resp.raise_for_status()
                results = resp.json()

                products = []
                for idx, product in enumerate(results):
                    product_id = product["product_id"]
                    product_name = product["product_name"]
                    shop_id = product["shop_id"]
                    price = product["price"]

                    products.append(f"{idx+1}. {product_name}\nProduct ID: {product_id}\nShop ID: {shop_id}\nPrice: ${price}")

                if products:
                    tool_response = f'A product search for "{query}" found {len(products)} results:\n\n' + "\n\n".join(products)
                    
                else:
                    tool_response = f'Product search for "{query}" found no results.'
                break
            except Exception as e:
                print(f"Error occurred during product search: {e}, retry {i + 1}/{MAX_RETRIES}", file=sys.stderr)
                if i == MAX_RETRIES - 1:  # ✓ 只在最后一次更新错误消息
                    tool_response = f"Error occurred during product search: {e}"
                await asyncio.sleep(RETRY_DELAY) 

        self._instance_dict[instance_id]["response"] = tool_response

        reward = await self.calc_reward(instance_id)

        return ToolResponse(text=tool_response), reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
       
        try:
            entry = self._instance_dict.get(instance_id)
            if not entry or not entry.get("question_id") or "parameters" not in entry:
                return 0.0
            question_id = entry["question_id"]
            params = entry["parameters"]

            query = params.get("query")
            shop_id = params.get("shop_id")
            price = params.get("price")

            if not query:
                return 0.0

            kwargs = {"query": query}
            if shop_id:
                kwargs["shop_id"] = shop_id
            if price:
                kwargs["price"] = price

            payload = {
                "question_id": question_id,
                "name": "product_search",
                "kwargs": kwargs
            }

            resp = await self.reward_client.post("/reward", json=payload)
            if resp.status_code != 200:
                logger.warning(f"[ProductSearch] Reward server returned {resp.status_code}: {resp.text}")
                return 0.0

            data = resp.json()
            return float(data.get("reward", 0.0))

        except Exception as e:
            logger.error(f"[ProductSearch] Failed to calculate reward: {e}")
            return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class ProductView(BaseTool):
    """Tool for finding memories similar to a specific memory."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._client = get_shared_client(PRODUCT_BASE_URL, timeout=TIMEOUT)
        self.reward_client = get_shared_client(REWARD_BASE_URL, timeout=TIMEOUT)
        self._instance_dict = {}
    

    async def fetch_product_detail(self, product_ids: List[str]) -> List[str]:
        if not product_ids:
            return []
        product_ids = [
            s.strip() 
            for pid in product_ids 
            if pid is not None and (s := str(pid)).strip()
        ]

        results = []
        for i in range(MAX_RETRIES):
            try:
                url = f"/product_view?product_ids={','.join(product_ids)}"
                # async with httpx.AsyncClient() as client:
                #     resp = await client.get(url, timeout=TIMEOUT)
                #     resp.raise_for_status()
                #     results = resp.json()
                resp = await self._client.get(url)
                resp.raise_for_status()
                results = resp.json()

                d = {p["product_id"]: p for p in results}
    
                tool_responses = []
                for product_id in product_ids:
                    if product_id in d:
                        product = d[product_id]
                        attributes = product.get("attributes")
                        options = product.get("options")

                        attributes_str = ""
                        if attributes:
                            attributes_str = "; ".join([f"{k} = {', '.join(vs)}" for k, vs in attributes.items()])

                        options_str = ""
                        if options:
                            for option in options:
                                if option:
                                    options_str += "- " +"; ".join([f"{k} = {', '.join(vs)}" for k, vs in option.items()]) + "\n"

                        tool_responses.append(f"The detail of product {product_id} is as follows:\nAttributes: {attributes_str}\nOptions:\n{options_str}".strip())
                    else:
                        tool_responses.append(f"The detail of product {product_id} is not found.")
                return tool_responses
            except Exception as e:
                print(
                    f"Error occurred during fetching product detail: {e}, retry {i + 1}/{MAX_RETRIES}",
                    file=sys.stderr,
                )
                if i == MAX_RETRIES - 1:  
                    tool_responses = [f"Error occurred during fetching product detail: {e}"]
                await asyncio.sleep(RETRY_DELAY)
        return tool_responses

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", kwargs)
        question_id = create_kwargs.get("question_id")
        self._instance_dict[instance_id] = {
            "response": "",
            "question_id": question_id,
            }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        
        product_ids = parameters.get("product_ids")
        self._instance_dict[instance_id]["parameters"] = parameters

        if not product_ids:
            error_msg = "Error: 'product_ids' is required."
            logger.error(f"[product_view] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}
        if not isinstance(product_ids, list):
            error_msg = "Product IDs must be a list of strings."
            return ToolResponse(text=json.dumps({"error": error_msg})), -1.0, {}
        
        tool_responses = await self.fetch_product_detail(product_ids)

        delimiter = "\n\n" + "-" * 10 + "\n\n"
        tool_responses = delimiter.join(tool_responses).strip()
        self._instance_dict[instance_id]["response"] = tool_responses

        reward = await self.calc_reward(instance_id)

        return ToolResponse(text=tool_responses), reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
       
        try:
            entry = self._instance_dict.get(instance_id)
            if not entry or not entry.get("question_id") or "parameters" not in entry:
                return 0.0
            question_id = entry["question_id"]
            params = entry["parameters"]

            product_ids = params.get("product_ids")
            if not product_ids or not isinstance(product_ids, list):
                return 0.0

            # 过滤有效 product_id
            valid_ids = [str(pid) for pid in product_ids if pid and isinstance(pid, (str, int))]
            if not valid_ids:
                return 0.0

            payload = {
                "question_id": question_id,
                "name": "product_view",
                "kwargs": {"product_ids": valid_ids}
            }

            resp = await self.reward_client.post("/reward", json=payload)
            if resp.status_code != 200:
                logger.warning(f"[ProductView] Reward server returned {resp.status_code}: {resp.text}")
                return 0.0

            data = resp.json()
            return float(data.get("reward", 0.0))

        except Exception as e:
            logger.error(f"[ProductView] Failed to calculate reward: {e}")
            return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class WebSearch(BaseTool):
    """Tool for finding memories similar to a specific memory."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def parse_google_search_page(self, idx: int, page: dict) -> str:
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


    async def google_search(self, query: str) -> str:
        url = f"{AI_HUB_SEARCH_BASE_URL}/customsearch/google/search"
        headers = {
            "Authorization": f"Bearer {AI_HUB_SEARCH_TOKEN}",
            "Content-Type": "application/json",
        }
        body = {"q": query}
        tool_response = f"Error occurred during web search."  # ✓ 初始化

        for i in range(MAX_RETRIES):
            try:
                # async with httpx.AsyncClient() as client:
                #     response = await client.post(url, headers=headers, json=body, timeout=TIMEOUT)
                #     response.raise_for_status()
                #     results = response.json()
                response = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
                response.raise_for_status()
                results = response.json()
                idx = 0
                web_snippets = []
                if "organic" in results:
                    for page in results["organic"]:
                        idx += 1
                        web_snippets.append(self.parse_google_search_page(idx, page))
                else:
                    for value in results.values():
                        if isinstance(value, list) and len(value) > 0:
                            for page in value:
                                if isinstance(page, dict):
                                    idx += 1
                                    web_snippets.append(self.parse_google_search_page(idx, page))
                if web_snippets:
                    tool_response = f'A web search for "{query}" found {len(web_snippets)} results:\n\n' + "\n\n".join(web_snippets)
                    break
                else:
                    tool_response = f'Web search for "{query}" found no results.'
            except Exception as e:
                print(f"Error occurred during web search: {e}, retry {i + 1}/{MAX_RETRIES}", file=sys.stderr)
                tool_response = f"Error occurred during web search: {e}"
                if i < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)

        return tool_response


    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        
        queries = parameters.get("queries")

        if not queries:
            error_msg = "Error: 'queries' is required."
            logger.error(f"[web_search] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}
        if not isinstance(queries, list):
            error_msg = "queries must be a list of strings."
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}
        
        tasks = [self.google_search(query) for query in queries]
        tool_responses = await asyncio.gather(*tasks)

        delimiter = "\n\n" + "-" * 10 + "\n\n"
        tool_responses = delimiter.join(tool_responses).strip()
        self._instance_dict[instance_id]["response"] = tool_responses
        return ToolResponse(text=tool_responses), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
       
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class WebVisit(BaseTool):
    """Tool for finding memories similar to a specific memory."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def truncate_webpage_content(self, content: str) -> str:
        encoding = tiktoken.get_encoding("o200k_base")

        tokens = encoding.encode(content)
        if len(tokens) <= MAX_TOKENS:
            return content

        truncated_tokens = tokens[:MAX_TOKENS]
        return encoding.decode(truncated_tokens)


    async def batch_summarize_webpage_content(self, urls: List[str], contents: List[str], goal: str) -> str:
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
        llm_responses = await asyncio.to_thread(
            client.batch_complete, 
            llm_requests
        )

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

        template = 'The useful information in "{url}" for goal "{goal}" is as follows: \n\n'
        template += "**Evidence in page**: \n{evidence}\n\n"
        template += "**Summary**: \n{summary}"
        evidence = "The provided webpage content could not be accessed. Please check the URL or file format."
        summary = "The webpage content could not be processed, and therefore, no information is available."
        tool_responses = []
        for i, url in enumerate(urls):
            if i in idx2summary:
                tool_responses.append(template.format(url=url, goal=goal, evidence=idx2summary[i]["evidence"], summary=idx2summary[i]["summary"]))
            else:
                tool_responses.append(template.format(url=url, goal=goal, evidence=evidence, summary=summary))
        delimiter = "\n\n" + "-" * 10 + "\n\n"
        return delimiter.join(tool_responses).strip()

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
        }
        return instance_id, ToolResponse()

    async def batch_fetch_webpage_content(self, urls: List[str]) -> List[str]:
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
                    await asyncio.sleep(RETRY_DELAY)
            return ""

        tasks = [_fetch_single(target_url) for target_url in urls]
        results = await asyncio.gather(*tasks)
        return results


    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        
        urls = parameters.get("urls")
        goal = parameters.get("goal")

        if not urls:
            error_msg = "Error: 'urls' is required."
            logger.error(f"[web_visit] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}
        if not isinstance(urls, list):
            error_msg = "urls must be a list of strings."
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}
        if not goal:
            error_msg = "Error: 'goal' is required."
            logger.error(f"[web_visit] {error_msg}")
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}
        
        contents = await self.batch_fetch_webpage_content(urls)
        contents = [self.truncate_webpage_content(content) for content in contents]
        tool_responses = await self.batch_summarize_webpage_content(urls, contents, goal)

        self._instance_dict[instance_id]["response"] = tool_responses
        return ToolResponse(text=tool_responses), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
       
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
