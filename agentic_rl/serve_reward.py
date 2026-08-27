# File: src/serve_reward.py

import os
import sys
from typing import Optional, Dict, Any, Iterable
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import ujson as json
from tqdm import tqdm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer
from pyserini.search.lucene import LuceneSearcher
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from util.misc import convert_date_to_timestamp
from mem.retriever import Retriever, ConversationDTO
from util.llm import ParallelOpenAICompletion, CompletionRequest
from prompt.evaluate import single_product_evaluation_prompt, add_on_deals_evaluation_prompt


# ================================
# Data Models
# ================================

class RewardRequest(BaseModel):
    question_id: str
    name: str
    kwargs: Dict[str, Any]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ================================
# Core Logic Class
# ================================

class RewardServer:
    """
    Core reward calculation logic. Can be attached to app.state.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self):
        self.searcher: Optional[LuceneSearcher] = None
        self.retrievers: Dict[str, Retriever] = {}
        self.references: Dict[str, ConversationDTO] = {}
        self.client: Optional[ParallelOpenAICompletion] = None
        self.thread_pool: Optional[ThreadPoolExecutor] = None

    def init_llm_client(self, base_url: str = None, api_key: str = None):
        """初始化 LLM 客户端用于商品评估"""
        self.client = ParallelOpenAICompletion(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_workers=100,
            max_retries=3,
            retry_delay=1,
        )
        print("✅ LLM client initialized.", file=sys.stderr)

    def load_searcher(self, index_dir: str):
        print(f"🔍 Loading product searcher from {index_dir}", file=sys.stderr)
        self.searcher = LuceneSearcher(index_dir)
        print(f"✅ Loaded product searcher.", file=sys.stderr)

    def load_retriever(self, index_dir: str, sentence_model_name: str):
        print(f"🔍 Loading memory retrievers from {index_dir}...", file=sys.stderr)
        sentence_model = SentenceTransformer(sentence_model_name)
        loaded = 0
        for conversation_id in os.listdir(index_dir):
            full_path = os.path.join(index_dir, conversation_id)
            if not os.path.isdir(full_path):
                continue
            try:
                retriever = Retriever(sentence_model=sentence_model)
                retriever.load_index(index_dir=index_dir, conversation_id=conversation_id)
                self.retrievers[conversation_id] = retriever
                loaded += 1
            except Exception as e:
                print(f"⚠️ Skip loading retriever {conversation_id}: {e}", file=sys.stderr)
        print(f"✅ Loaded {loaded} memory retrievers.", file=sys.stderr)

    def load_references(self, reference_file: str):
        print(f"🔍 Loading references from {reference_file}...", file=sys.stderr)
        with open(reference_file, "r") as fin:
            for line in tqdm(fin, desc="Loading references"):
                try:
                    item = json.loads(line.strip())
                    conversation = ConversationDTO(**item)
                    self.references[conversation.question_id] = conversation
                except Exception as e:
                    print(f"⚠️ Skip invalid line: {e}", file=sys.stderr)
        print(f"✅ Loaded {len(self.references)} reference conversations.", file=sys.stderr)

    # ============================================
    # Reward Calculation Methods
    # ============================================

    def calc_reward_mem_search(self, question_id: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """计算内存搜索奖励"""
        if question_id not in self.references:
            return None
        reference = self.references[question_id]

        conversation_id = question_id.rsplit("_", 1)[0]
        retriever = self.retrievers.get(conversation_id)
        if not retriever:
            return None

        queries = kwargs.get("queries")
        if not queries or not isinstance(queries, list):
            return None

        result = {"total": 0, "hit": 0, "reward": 0.0}
        for query in queries:
            indices, _ = retriever.search(query=query, k=10)
            q_total = 0
            q_hit = 0
            for idx in indices:
                session_idx = retriever.idx2sess[idx]
                session_id = reference.haystack_session_ids[session_idx]
                if session_id in reference.answer_session_ids:
                    q_hit += 1
                q_total += 1
            q_reward = q_hit / q_total if q_total > 0 else 0
            result[query] = {"total": q_total, "hit": q_hit, "reward": q_reward}
            result["total"] += 1
            result["hit"] += q_reward
        result["reward"] = result["hit"] / result["total"] if result["total"] > 0 else 0.0
        return result

    def calc_reward_mem_view(self, question_id: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """计算内存查看奖励"""
        if question_id not in self.references:
            return None
        reference = self.references[question_id]

        conversation_id = question_id.rsplit("_", 1)[0]
        retriever = self.retrievers.get(conversation_id)
        if not retriever:
            return None

        indices = kwargs.get("indices")
        if not indices or not isinstance(indices, list):
            return None

        result = {"total": 0, "hit": 0, "reward": 0.0}
        for idx in indices:
            session_idx = retriever.idx2sess[idx]
            session_id = reference.haystack_session_ids[session_idx]
            if session_id in reference.answer_session_ids:
                result["hit"] += 1
            result["total"] += 1
        result["reward"] = result["hit"] / result["total"] if result["total"] > 0 else 0.0
        return result

    def calc_reward_mem_summarize_by_date(self, question_id: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """计算日期范围总结奖励"""
        if question_id not in self.references:
            return None
        reference = self.references[question_id]

        start_date = kwargs.get("start_date")
        offset = kwargs.get("offset")
        goal = kwargs.get("goal")

        if not start_date or not offset or not goal:
            return None
        
        try:
            start_timestamp = convert_date_to_timestamp(start_date)
        except Exception:
            return None

        try:
            offset = int(offset)
            if offset < 1 or offset > 7:
                return None
        except ValueError:
            return None

        if not isinstance(goal, str):
            return None

        end_timestamp = start_timestamp + offset * 86400  # 24*60*60

        result = {"total": 0, "hit": 0, "reward": 0.0}
        for answer_session_id in reference.answer_session_ids:
            try:
                ind = reference.haystack_session_ids.index(answer_session_id)
                date = reference.haystack_dates[ind]
                timestamp = convert_date_to_timestamp(date)
                if start_timestamp <= timestamp <= end_timestamp:
                    result["hit"] += 1
            except (ValueError, IndexError):
                pass
            result["total"] += 1
        
        result["reward"] = result["hit"] / result["total"] if result["total"] > 0 else 0.0
        return result

    def calc_reward_product_search(self, question_id: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """计算商品搜索奖励（支持 LLM 评估）"""
        if question_id not in self.references or not self.searcher:
            return None
        reference = self.references[question_id]

        query = kwargs.get("query")
        shop_id = kwargs.get("shop_id")
        price = kwargs.get("price")

        if not query or not isinstance(query, str):
            return None

        price_min = price_max = None
        if price:
            try:
                parts = price.split("-")
                low = parts[0].strip() if parts[0] else ""
                high = parts[1].strip() if len(parts) > 1 and parts[1] else ""
                price_min = float(low) if low else 0.0
                price_max = float(high) if high else float("inf")
                assert price_min >= 0 and price_min <= price_max
            except Exception:
                return None

        # 搜索范围：如果有过滤条件，搜索更多
        k = 50 if not shop_id and not price else 10000
        docs = self.searcher.search(q=query, k=k, remove_dups=True)
        product_ids = set()

        for doc in docs:
            try:
                raw_data = self.searcher.doc(doc.docid)
                if not raw_data:
                    continue
                product = json.loads(raw_data.raw())["product"]
                pid = product["product_id"]
                sid = product["seller_id"]
                pprice = product["price"]

                if shop_id and shop_id != sid:
                    continue
                if price_min is not None and (pprice < price_min or pprice > price_max):
                    continue
                product_ids.add(pid)
                if len(product_ids) >= 50:
                    break
            except Exception as e:
                print(f"Error parsing doc {doc.docid}: {e}", file=sys.stderr)

        result = {"total": 0, "hit": 0, "reward": 0.0}
        ans = reference.answer
        
        if reference.question_type == "single_product":
            result["total"] = 1
            answer_pid = ans.get("product_id")
            if answer_pid in product_ids:
                result["hit"] = 1
            elif self.client and ans.get("wanted_features"):
                # 使用 LLM 评估
                is_match = self._evaluate_single_product(
                    reference.question,
                    ans.get("wanted_features", []),
                    product_ids
                )
                result["hit"] = int(is_match)
            else:
                result["hit"] = 0
                
        elif reference.question_type == "add_on_deals":
            for pref in ans.get("preferences", []):
                result["total"] += 1
                pref_pid = pref.get("product_id")
                if pref_pid in product_ids:
                    result["hit"] += 1
                elif self.client and pref.get("wanted_features"):
                    # 使用 LLM 评估
                    is_match = self._evaluate_single_product(
                        reference.question,
                        pref.get("wanted_features", []),
                        product_ids
                    )
                    result["hit"] += int(is_match)
        
        result["reward"] = result["hit"] / result["total"] if result["total"] > 0 else 0.0
        return result

    def calc_reward_product_view(self, question_id: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """计算商品查看奖励（支持 LLM 评估）"""
        if question_id not in self.references:
            return None
        reference = self.references[question_id]

        product_ids = kwargs.get("product_ids")
        if not product_ids or not isinstance(product_ids, list):
            return None
        given_ids = set(map(str, product_ids))

        result = {"total": 0, "hit": 0, "reward": 0.0}
        ans = reference.answer
        
        if reference.question_type == "single_product":
            result["total"] = 1
            answer_pid = ans.get("product_id")
            if answer_pid in given_ids:
                result["hit"] = 1
            elif self.client and ans.get("wanted_features"):
                # 使用 LLM 评估
                is_match = self._evaluate_single_product(
                    reference.question,
                    ans.get("wanted_features", []),
                    given_ids
                )
                result["hit"] = int(is_match)
            else:
                result["hit"] = 0
                
        elif reference.question_type == "add_on_deals":
            for pref in ans.get("preferences", []):
                result["total"] += 1
                pref_pid = pref.get("product_id")
                if pref_pid in given_ids:
                    result["hit"] += 1
                elif self.client and pref.get("wanted_features"):
                    # 使用 LLM 评估
                    is_match = self._evaluate_single_product(
                        reference.question,
                        pref.get("wanted_features", []),
                        given_ids
                    )
                    result["hit"] += int(is_match)
        
        result["reward"] = result["hit"] / result["total"] if result["total"] > 0 else 0.0
        return result

    # ============================================
    # LLM Evaluation Helper
    # ============================================

    def _evaluate_single_product(
        self,
        question: str,
        wanted_features: Iterable[str],
        product_ids: Iterable[str]
    ) -> bool:
        """
        使用 LLM 评估推荐商品是否满足用户需求。
        
        Args:
            question: 用户问题
            wanted_features: 需要的特性列表
            product_ids: 推荐商品 ID 列表
        
        Returns:
            bool: 至少一个商品满足需求
        """
        if not self.client or not self.searcher:
            return False

        llm_requests = []
        
        for product_id in product_ids:
            try:
                raw_product = self.searcher.doc(product_id)
                if not raw_product:
                    continue
                
                product = json.loads(raw_product.raw())["product"]
                product_name = product.get("product_name", "")
                attributes = product.get("attributes", {})
                options = product.get("options", [])

                # 格式化属性
                attributes_str = ""
                if attributes:
                    attributes_str = "; ".join([
                        f"{k} = {', '.join(vs)}"
                        for k, vs in sorted(attributes.items(), key=lambda x: x[0])
                    ])

                # 格式化选项
                options_str = ""
                if options:
                    for option in options:
                        if option:
                            options_str += "- " + "; ".join([
                                f"{k} = {', '.join(vs)}"
                                for k, vs in sorted(option.items(), key=lambda x: x[0])
                            ]) + "\n"

                recommended_product = f"Product Name: {product_name}\nAttributes: {attributes_str}\nOptions:\n{options_str}".strip()

                # 格式化需要的特性
                wanted_features_str = ""
                for feature in wanted_features:
                    wanted_features_str += f"- {feature}\n"

                # 构建提示词
                prompt = single_product_evaluation_prompt.format(
                    user_query=question,
                    wanted_features=wanted_features_str,
                    recommended_product=recommended_product,
                )
                
                llm_requests.append(
                    CompletionRequest(
                        messages=[{"role": "user", "content": prompt}],
                        model="gpt-4-turbo",
                        extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                    )
                )
            except Exception as e:
                print(f"Error processing product {product_id}: {e}", file=sys.stderr)
                continue

        if not llm_requests:
            return False

        # 批量调用 LLM
        llm_responses = self.client.batch_complete(llm_requests, verbose=False)
        
        for llm_response in llm_responses:
            if llm_response.success:
                if "yes" in llm_response.content.strip().lower():
                    return True
            else:
                print(f"LLM error: {llm_response.error}", file=sys.stderr)
        
        return False


# ================================
# FastAPI App Setup
# ================================

app = FastAPI(
    title="Reward Server API",
    description="Calculate retrieval rewards for memory and product tasks.",
    version="1.0.0"
)


@app.on_event("startup")
def startup_event():
    """应用启动时初始化所有资源"""
    # 设置环境变量
    os.environ.setdefault('MULTIPROCESS_RESOURCE_TRACKING', '0')

    # 读取环境变量或使用默认值
    product_index_dir = os.getenv("PRODUCT_INDEX_DIR", "data/product_indexes")
    mem_index_dir = os.getenv("MEM_INDEX_DIR", "data/mem_indexes")
    reference_file = os.getenv("REFERENCE_FILE", "data/shopping_companion_s_cleaned.jsonl")
    sentence_model_name = os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    worker_threads = int(os.getenv("WORKER_THREADS", multiprocessing.cpu_count() * 2))
    enable_llm = os.getenv("ENABLE_LLM_EVALUATION", "false").lower() == "true"

    print("🚀 Starting up Reward Server...", file=sys.stderr)

    # 初始化 RewardServer 实例并挂载到 app.state
    server = RewardServer()
    server.load_searcher(product_index_dir)
    server.load_retriever(mem_index_dir, sentence_model_name)
    server.load_references(reference_file)
    
    # 可选：初始化 LLM 客户端
    if enable_llm and openai_api_key:
        server.init_llm_client(base_url=openai_base_url, api_key=openai_api_key)
    
    server.thread_pool = ThreadPoolExecutor(max_workers=worker_threads)
    app.state.reward_server = server

    print("✅ Reward Server is ready!", file=sys.stderr)
    print(f"   📊 Loaded {len(server.references)} references", file=sys.stderr)
    print(f"   🔍 Loaded {len(server.retrievers)} memory retrievers", file=sys.stderr)
    print(f"   🤖 LLM evaluation: {'enabled' if enable_llm else 'disabled'}", file=sys.stderr)


@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时清理资源"""
    server = getattr(app.state, 'reward_server', None)
    if server and hasattr(server, "thread_pool") and server.thread_pool:
        server.thread_pool.shutdown(wait=True)
        print("✅ Thread pool shutdown complete.", file=sys.stderr)
    print("👋 Reward Server shutdown complete.", file=sys.stderr)


# ================================
# API Routes
# ================================

@app.get("/")
def index():
    """健康检查端点"""
    return {
        "service": "Reward Server API",
        "version": "1.0.0",
        "status": "running",
        "usage": "POST /reward",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """健康检查端点（带详细信息）"""
    server = app.state.reward_server
    return {
        "status": "healthy",
        "references_loaded": len(server.references),
        "retrievers_loaded": len(server.retrievers),
        "searcher_ready": server.searcher is not None,
        "llm_enabled": server.client is not None,
    }


@app.post("/reward")
def calc_reward(request: RewardRequest):
    """
    计算奖励值
    
    支持的奖励类型：
    - mem_search: 内存搜索奖励
    - mem_view: 内存查看奖励
    - mem_summarize_by_date: 日期范围总结奖励
    - product_search: 商品搜索奖励
    - product_view: 商品查看奖励
    """
    server = app.state.reward_server

    try:
        if request.name == "mem_search":
            result = server.calc_reward_mem_search(request.question_id, request.kwargs)
        elif request.name == "mem_view":
            result = server.calc_reward_mem_view(request.question_id, request.kwargs)
        elif request.name == "mem_summarize_by_date":
            result = server.calc_reward_mem_summarize_by_date(request.question_id, request.kwargs)
        elif request.name == "product_search":
            result = server.calc_reward_product_search(request.question_id, request.kwargs)
        elif request.name == "product_view":
            result = server.calc_reward_product_view(request.question_id, request.kwargs)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown reward type: {request.name}. "
                        f"Supported: mem_search, mem_view, mem_summarize_by_date, "
                        f"product_search, product_view"
            )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Question ID not found or invalid input parameters."
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error calculating reward for {request.question_id}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/reward/batch")
def calc_reward_batch(requests: list[RewardRequest]):
    """
    批量计算奖励值
    
    返回列表，对应输入的顺序
    """
    server = app.state.reward_server
    results = []
    errors = []

    for i, req in enumerate(requests):
        try:
            if req.name == "mem_search":
                result = server.calc_reward_mem_search(req.question_id, req.kwargs)
            elif req.name == "mem_view":
                result = server.calc_reward_mem_view(req.question_id, req.kwargs)
            elif req.name == "mem_summarize_by_date":
                result = server.calc_reward_mem_summarize_by_date(req.question_id, req.kwargs)
            elif req.name == "product_search":
                result = server.calc_reward_product_search(req.question_id, req.kwargs)
            elif req.name == "product_view":
                result = server.calc_reward_product_view(req.question_id, req.kwargs)
            else:
                result = None
                errors.append({"index": i, "error": f"Unknown reward type: {req.name}"})

            if result is None and not errors or (errors and errors[-1]["index"] != i):
                errors.append({"index": i, "error": "Question ID not found or invalid input"})

            results.append(result)
        except Exception as e:
            print(f"❌ Error in batch request {i}: {e}", file=sys.stderr)
            results.append(None)
            errors.append({"index": i, "error": str(e)})

    return {
        "results": results,
        "errors": errors,
        "total": len(requests),
        "success_count": sum(1 for r in results if r is not None)
    }


# ================================
# 启动命令
# ================================



# nohup uvicorn agentic_rl.serve_reward:app --host 0.0.0.0 --port 5633 --workers 2 --limit-concurrency 1000 --timeout-keep-alive 30 --log-level info > logs/serve_reward.log 2>&1 &
