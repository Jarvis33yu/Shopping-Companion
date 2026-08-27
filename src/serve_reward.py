import os
import sys
import argparse
import multiprocessing
from typing import Dict, Any, Optional, Iterable

import ujson as json
from tqdm import tqdm
from flask import Flask, request, jsonify
from waitress import serve
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from pyserini.search.lucene import LuceneSearcher

from util.misc import convert_date_to_timestamp
from mem.retriever import Retriever, ConversationDTO
from util.llm import ParallelOpenAICompletion, CompletionRequest
from prompt.evaluate import single_product_evaluation_prompt, add_on_deals_evaluation_prompt


app = Flask(__name__)
server = None


class RewardServer(BaseModel):
    searcher: Any = None
    retrievers: Dict[str, Retriever] = {}
    references: Dict[str, ConversationDTO] = {}
    client: Any = ParallelOpenAICompletion(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_workers=100,
        max_retries=3,
        retry_delay=1,
    )

    def load_searcher(self, index_dir: str):
        print(f"Loading searcher from {index_dir}", file=sys.stderr)
        self.searcher = LuceneSearcher(index_dir)
        print(f"Loaded searcher from {index_dir}.", file=sys.stderr)

    def load_retriever(self, index_dir: str, sentence_model_name: str):
        sentence_model = SentenceTransformer(sentence_model_name)
        for conversation_id in tqdm(os.listdir(index_dir), desc="Loading retrievers: "):
            retriever = Retriever(sentence_model=sentence_model)
            retriever.load_index(index_dir=index_dir, conversation_id=conversation_id)
            self.retrievers[conversation_id] = retriever
        print(f"Loaded retrievers from {index_dir}.", file=sys.stderr)

    def load_references(self, reference_file: str):
        with open(reference_file, "r") as fin:
            for line in tqdm(fin, desc="Loading references: "):
                item = json.loads(line.strip())
                conversation = ConversationDTO(**item)
                self.references[conversation.question_id] = conversation
        print(f"Loaded references from {reference_file}.", file=sys.stderr)

    def calc_reward_mem_search(self, question_id, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if question_id not in self.references:
            return None
        reference = self.references[question_id]

        conversation_id = question_id.rsplit("_", 1)[0]
        if conversation_id not in self.retrievers:
            return None
        retriever = self.retrievers[conversation_id]

        queries = kwargs.get("queries")
        if not queries:
            return None
        if not isinstance(queries, list):
            return None

        result = {
            "total": 0,
            "hit": 0,
            "reward": 0.0,
        }
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
            result[query] = {
                "total": q_total,
                "hit": q_hit,
                "reward": q_hit / q_total,
            }
            result["total"] += 1
            result["hit"] += q_hit / q_total if q_total > 0 else 0
        result["reward"] = result["hit"] / result["total"]
        return result

    def calc_reward_mem_view(self, question_id, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if question_id not in self.references:
            return None
        reference = self.references[question_id]

        conversation_id = question_id.rsplit("_", 1)[0]
        if conversation_id not in self.retrievers:
            return None
        retriever = self.retrievers[conversation_id]

        indices = kwargs.get("indices")
        if not indices:
            return None
        if not isinstance(indices, list):
            return None

        result = {
            "total": 0,
            "hit": 0,
            "reward": 0.0,
        }
        for idx in indices:
            session_idx = retriever.idx2sess[idx]
            session_id = reference.haystack_session_ids[session_idx]
            if session_id in reference.answer_session_ids:
                result["hit"] += 1
            result["total"] += 1
        result["reward"] = result["hit"] / result["total"]
        return result

    def calc_reward_mem_summarize_by_date(self, question_id, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if question_id not in self.references:
            return None
        reference = self.references[question_id]

        start_date = kwargs.get("start_date")
        offset = kwargs.get("offset")
        goal = kwargs.get("goal")

        if not start_date:
            return None
        try:
            start_timestamp = convert_date_to_timestamp(start_date)
        except:
            return None

        if not offset:
            return None
        if not isinstance(offset, int) or (
            isinstance(offset, str) and not offset.isdigit()
        ):
            return None
        offset = int(offset)
        if offset < 1 or offset > 7:
            return None

        if not goal:
            return None
        if not isinstance(goal, str):
            return None

        end_timestamp = start_timestamp + offset * 24 * 60 * 60

        result = {
            "total": 0,
            "hit": 0,
            "reward": 0.0,
        }
        for answer_session_id in reference.answer_session_ids:
            ind = reference.haystack_session_ids.index(answer_session_id)
            date = reference.haystack_dates[ind]
            timestamp = convert_date_to_timestamp(date)
            if timestamp >= start_timestamp and timestamp <= end_timestamp:
                result["hit"] += 1
            result["total"] += 1
        result["reward"] = result["hit"] / result["total"]
        return result

    def calc_reward_product_search(self, question_id, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if question_id not in self.references:
            return None
        reference = self.references[question_id]

        query = kwargs.get("query")
        shop_id = kwargs.get("shop_id")
        price = kwargs.get("price")

        if not query:
            return None
        if not isinstance(query, str):
            return None

        price_min = price_max = None
        if price:
            try:
                price_range = price.split("-")
                price_min = float(price_range[0]) if price_range[0] else 0.0
                price_max = float(price_range[1]) if price_range[1] else float("inf")
                assert price_min <= price_max
                assert price_min >= 0
            except Exception as e:
                return None

        k = 50
        capacity = k if not shop_id and not price else 10000
        docs = self.searcher.search(q=query, k=capacity)
        product_ids = set()
        for doc in docs:
            raw_product = self.searcher.doc(doc.docid)
            if not raw_product:
                continue
            product = json.loads(raw_product.raw())["product"]
            product_id = product["product_id"]
            shop_id = product["seller_id"]
            price = product["price"]

            if shop_id and shop_id != product["seller_id"]:
                continue
            if price_min is not None and (price < price_min or price > price_max):
                continue
            product_ids.add(product_id)

            if len(product_ids) >= k:
                break

        result = {
            "total": 0,
            "hit": 0,
            "reward": 0.0,
        }
        if reference.question_type == "single_product":
            result["total"] = 1
            if reference.answer["product_id"] in product_ids:
                result["hit"] = 1
            else:
                result["hit"] = int(self._evaluate_single_product(reference.question, reference.answer["wanted_features"], product_ids))
        elif reference.question_type == "add_on_deals":
            for preference in reference.answer["preferences"]:
                result["total"] += 1
                if preference["product_id"] in product_ids:
                    result["hit"] += 1
                else:
                    result["hit"] += int(self._evaluate_single_product(reference.question, preference["wanted_features"], product_ids))
        result["reward"] = result["hit"] / result["total"]
        return result

    def calc_reward_product_view(self, question_id, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if question_id not in self.references:
            return None
        reference = self.references[question_id]

        product_ids = kwargs.get("product_ids")
        if not product_ids:
            return None
        if not isinstance(product_ids, list):
            return None
        product_ids = set(product_ids)

        result = {
            "total": 0,
            "hit": 0,
            "reward": 0.0,
        }
        if reference.question_type == "single_product":
            result["total"] = 1
            if reference.answer["product_id"] in product_ids:
                result["hit"] = 1
            else:
                result["hit"] = int(self._evaluate_single_product(reference.question, reference.answer["wanted_features"], product_ids))
        elif reference.question_type == "add_on_deals":
            for preference in reference.answer["preferences"]:
                result["total"] += 1
                if preference["product_id"] in product_ids:
                    result["hit"] += 1
                else:
                    result["hit"] += int(self._evaluate_single_product(reference.question, preference["wanted_features"], product_ids))
        result["reward"] = result["hit"] / result["total"]
        return result


    def _evaluate_single_product(self, question: str, wanted_features: Iterable[str], product_ids: Iterable[str]) -> bool:
        llm_requests = []
        for product_id in product_ids:
            raw_product = self.searcher.doc(product_id)
            if not raw_product:
                continue
            product = json.loads(raw_product.raw())["product"]
            product_name = product["product_name"]
            attributes = product.get("attributes", {})
            options = product.get("options", [])

            # recommended product
            product_name = product["product_name"]
            attributes = product.get("attributes", {})
            options = product.get("options", [])

            attributes_str = ""
            if attributes:
                attributes_str = "; ".join([f"{k} = {', '.join(vs)}" for k, vs in sorted(attributes.items(), key=lambda x: x[0])])

            options_str = ""
            if options:
                for option in options:
                    if option:
                        options_str += "- " +"; ".join([f"{k} = {', '.join(vs)}" for k, vs in sorted(option.items(), key=lambda x: x[0])]) + "\n"

            recommended_product = f"Product Name: {product_name}\nAttributes: {attributes_str}\nOptions:\n{options_str}".strip()

            # wanted features
            wanted_features_str = ""
            for feature in wanted_features:
                wanted_features_str += f"- {feature}\n"

            # user query
            user_query = question

            prompt = single_product_evaluation_prompt.format(
                user_query=user_query,
                wanted_features=wanted_features_str,
                recommended_product=recommended_product,
            )
            llm_requests.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-2025-08-07-GlobalStandard",
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                )
            )

        llm_responses = self.client.batch_complete(llm_requests, verbose=True)
        for llm_response in llm_responses:
            if llm_response.success:
                if "yes" == llm_response.content.strip():
                    return True
            else:
                print(llm_response.error, file=sys.stderr)
        return False


@app.route("/")
def index():
    result = {
        "reward": 0.0,
    }
    question_id = request.args.get("question_id")
    name = request.args.get("name")
    kwargs = request.args.get("kwargs")

    try:
        kwargs = json.loads(kwargs)
    except:
        return jsonify(result)

    if name == "mem_search":
        result = server.calc_reward_mem_search(question_id, kwargs)
    elif name == "mem_view":
        result = server.calc_reward_mem_view(question_id, kwargs)
    elif name == "mem_summarize_by_date":
        result = server.calc_reward_mem_summarize_by_date(question_id, kwargs)
    elif name == "product_search":
        result = server.calc_reward_product_search(question_id, kwargs)
    elif name == "product_view":
        result = server.calc_reward_product_view(question_id, kwargs)
    else:
        print(f"Unknown reward name: {name}", file=sys.stderr)

    return jsonify(result)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--product_index_dir", type=str, required=True)
    args.add_argument("--mem_index_dir", type=str, required=True)
    args.add_argument("--reference_file", type=str, required=True)
    args.add_argument("--sentence_model_name", type=str, default="all-MiniLM-L6-v2")
    args.add_argument("--host", type=str, default="0.0.0.0")
    args.add_argument("--port", type=int, default=5633)
    args = args.parse_args()

    server = RewardServer()
    server.load_searcher(args.product_index_dir)
    server.load_retriever(args.mem_index_dir, args.sentence_model_name)
    server.load_references(args.reference_file)

    cores = multiprocessing.cpu_count()

    serve(
        app,
        host=args.host,
        port=args.port,
        threads=cores,
        expose_tracebacks=True,
        channel_timeout=10,
        cleanup_interval=10,
    )
