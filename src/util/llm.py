import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

import ujson as json
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm


@dataclass
class CompletionRequest:
    """单个completion请求的数据结构"""

    messages: List[Dict[str, str]]
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    extra_kwargs: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None


@dataclass
class CompletionResult:
    """单个completion请求的结果"""

    success: bool
    reasoning_content: Optional[str] = None
    content: Optional[str] = None
    error: Optional[str] = None
    request_index: Optional[int] = None  # 原始请求的索引
    raw_response: Optional[ChatCompletion] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class ParallelOpenAICompletion:
    """线程并行调用OpenAI completion接口的工具类"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_workers: int = 10,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        初始化并行OpenAI客户端

        Args:
            api_key: OpenAI API密钥，如果不提供则从环境变量OPENAI_API_KEY读取
            base_url: API基础URL，用于自定义端点
            max_workers: 最大并发线程数
            timeout: 单个请求的超时时间（秒）
            max_retries: 失败重试次数
            retry_delay: 重试延迟时间（秒）
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.base_url = base_url
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _call_completion(
        self,
        request: CompletionRequest,
        request_index: Optional[int] = None,
    ) -> CompletionResult:
        """
        调用单个completion请求（带重试机制）

        Args:
            request: completion请求对象
            request_index: 请求的索引，用于保持结果顺序

        Returns:
            CompletionResult对象
        """
        for attempt in range(self.max_retries):
            try:
                # 构建API调用参数
                kwargs = {
                    "model": request.model,
                    "messages": request.messages,
                }

                if request.temperature is not None:
                    kwargs["temperature"] = request.temperature
                if request.max_tokens is not None:
                    kwargs["max_tokens"] = request.max_tokens
                if request.max_completion_tokens is not None:
                    kwargs["max_completion_tokens"] = request.max_completion_tokens
                if request.extra_kwargs is not None:
                    kwargs.update(request.extra_kwargs)
                if request.response_format is not None:
                    kwargs["response_format"] = request.response_format
                if request.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = request.reasoning_effort

                # 调用API
                response = self.client.chat.completions.create(**kwargs)

                # 提取内容
                try:
                    reasoning_content = response.choices[0].message.reasoning_content
                except:
                    reasoning_content = None

                try:
                    content = response.choices[0].message.content
                except:
                    content = None

                usage = getattr(response, "usage", None)
                prompt_tokens = (
                    getattr(usage, "prompt_tokens", None) if usage is not None else None
                )
                completion_tokens = (
                    getattr(usage, "completion_tokens", None)
                    if usage is not None
                    else None
                )

                if not reasoning_content and not content:
                    raise ValueError("Reasoning content and content are both empty")

                return CompletionResult(
                    success=True,
                    reasoning_content=reasoning_content,
                    content=content,
                    request_index=request_index,
                    raw_response=response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            except Exception as e:
                error_msg = str(e)
                if attempt < self.max_retries - 1:
                    # 等待后重试
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    # 最后一次尝试失败
                    return CompletionResult(
                        success=False,
                        error=error_msg,
                        request_index=request_index,
                    )

        # 理论上不会到达这里
        return CompletionResult(
            success=False,
            error="Unknown error",
            request_index=request_index,
        )

    def batch_complete(
        self,
        requests: List[CompletionRequest],
        preserve_order: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verbose: bool = False,
    ) -> List[CompletionResult]:
        """
        批量并行调用completion接口

        Args:
            requests: completion请求列表
            preserve_order: 是否保持结果顺序（与输入顺序一致）
            progress_callback: 进度回调函数，参数为(已完成数, 总数)

        Returns:
            CompletionResult列表
        """
        if not requests:
            return []

        results = []
        total = len(requests)
        completed = 0

        pbar = tqdm(total=total, desc="Batch completing: ", disable=not verbose)
        if preserve_order:
            # 保持顺序：使用索引映射
            results = [None] * total

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_index = {
                    executor.submit(self._call_completion, req, idx): idx
                    for idx, req in enumerate(requests)
                }

                # 收集结果
                for future in as_completed(future_to_index):
                    pbar.update(1)
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
                    except Exception as e:
                        results[index] = CompletionResult(
                            success=False,
                            error=f"Future exception: {str(e)}",
                            request_index=index,
                        )
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
        else:
            # 不保持顺序：直接收集结果
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._call_completion, req, idx)
                    for idx, req in enumerate(requests)
                ]

                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
                    except Exception as e:
                        results.append(
                            CompletionResult(
                                success=False,
                                error=f"Future exception: {str(e)}",
                            )
                        )
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)

        return results


def parse_json_response(response: str) -> Dict[str, Any]:
    """解析JSON响应"""
    response_cleaned = response.strip().replace("```json", "").replace("```", "")
    try:
        # Clean the response in case there's extra text
        return json.loads(response_cleaned)
    except Exception as e:
        print(f"JSON parsing error in loads: {e}\nThe response is: {response}")

    try:
        # Try to find JSON content if wrapped in other text
        if not response_cleaned.startswith("{"):
            start_idx = response_cleaned.find("{")
            if start_idx != -1:
                response_cleaned = response_cleaned[start_idx:]
        if not response_cleaned.endswith("}"):
            end_idx = response_cleaned.rfind("}")
            if end_idx != -1:
                response_cleaned = response_cleaned[: end_idx + 1]

        return json.loads(response_cleaned)
    except Exception as e:
        print(f"JSON parsing error in find '{{': {e}\nThe response is: {response}")

    try:
        # Try to find JSON arry content if wrapped in other text
        if not response_cleaned.startswith("["):
            start_idx = response_cleaned.find("[")
            if start_idx != -1:
                response_cleaned = response_cleaned[start_idx:]
        if not response_cleaned.endswith("]"):
            end_idx = response_cleaned.rfind("]")
            if end_idx != -1:
                response_cleaned = response_cleaned[: end_idx + 1]

        return json.loads(response_cleaned)
    except Exception as e:
        print(f"JSON parsing error in find '[': {e}\nThe response is: {response}")

    return {}


if __name__ == "__main__":
    # 初始化客户端
    client = ParallelOpenAICompletion(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_workers=10,  # 并发线程数
        max_retries=3,  # 重试次数
    )

    # 准备多个请求
    requests = [
        CompletionRequest(
            messages=[{"role": "user", "content": "Hello!"}],
            model="qwen3-max",
            temperature=0.0,
            max_tokens=8192,
            extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
        ),
        CompletionRequest(
            messages=[{"role": "user", "content": "How are you?"}],
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=8192,
            extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
        ),
    ]

    # 批量并行调用
    results = client.batch_complete(requests, preserve_order=True)

    # 处理结果
    for result in results:
        if result.success:
            print(f"Content: {result.content}")
        else:
            print(f"Error: {result.error}")
