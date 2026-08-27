import os
import re
import sys
import argparse
from typing import Dict, List, Any, Tuple
from collections import Counter

import ujson as json
from tqdm import tqdm
from pydantic import BaseModel
from pyserini.search.lucene import LuceneSearcher

from mem.retriever import ConversationDTO
from prompt.evaluate import (
    single_product_evaluation_prompt,
    add_on_deals_evaluation_prompt,
    repeat_purchase_evaluation_prompt,
    complement_recognition_prompt,
    complement_evaluation_prompt,
)
from util.llm import ParallelOpenAICompletion, CompletionRequest


class Evaluator(BaseModel):
    args: Any

    searcher: Any = None
    client: Any = None
    id2conv: Dict[str, ConversationDTO] = {}
    id2trej: Dict[str, List[Dict]] = {}

    def evaluate(self):
        # Load searcher
        self.searcher = LuceneSearcher(self.args.index_dir)

        # Load client
        self.client = ParallelOpenAICompletion(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=self.args.max_workers,
            max_retries=3,
            retry_delay=1,
        )

        # Load id2conv
        with open(self.args.ref_file, "r") as fin:
            for line in tqdm(fin, desc="Loading id2conv: "):
                item = json.loads(line.strip())
                conversation = ConversationDTO(**item)

                self.id2conv[conversation.question_id] = conversation

        # 收集所有需要评估的任务，按类型分组
        tasks_by_type = {
            "single_product": [],
            "add_on_deals": [],
            "repeat_purchase": [],
            "complement": [],
        }

        with open(self.args.hyp_file, "r") as fin:
            for line in fin:
                item = json.loads(line.strip())

                if not item:
                    continue

                question_id = item[0].get("extra_info", {}).get("question_id")
                if not question_id:
                    continue

                if question_id not in self.id2conv:
                    continue
                conversation = self.id2conv[question_id]

                if not conversation.answer:
                    print(
                        f"No answer for conversation: {question_id}",
                        file=sys.stderr,
                    )
                    continue

                question_type = item[0].get("extra_info", {}).get("question_type")

                if question_type in tasks_by_type:
                    tasks_by_type[question_type].append((conversation, item))
                    self.id2trej[question_id] = item
                else:
                    print(f"Unknown question type: {question_type}", file=sys.stderr)

        # 按类型批量评估
        metrics = []

        # 评估 single_product 类型（最常见且最耗时的类型）- 使用批量方法
        if tasks_by_type["single_product"]:
            print(
                f"Evaluating {len(tasks_by_type['single_product'])} single_product tasks with batch mode..."
            )
            batch_metrics = self.batch_evaluate_single_product(
                tasks_by_type["single_product"]
            )
            metrics.extend(batch_metrics)
        if tasks_by_type["add_on_deals"]:
            print(
                f"Evaluating {len(tasks_by_type['add_on_deals'])} add_on_deals tasks with batch mode..."
            )
            batch_metrics = self.batch_evaluate_add_on_deals(
                tasks_by_type["add_on_deals"]
            )
            metrics.extend(batch_metrics)
        if tasks_by_type["repeat_purchase"]:
            print(
                f"Evaluating {len(tasks_by_type['repeat_purchase'])} repeat_purchase tasks with batch mode..."
            )
            batch_metrics = self.batch_evaluate_repeat_purchase(
                tasks_by_type["repeat_purchase"]
            )
            metrics.extend(batch_metrics)
        if tasks_by_type["complement"]:
            print(
                f"Evaluating {len(tasks_by_type['complement'])} complement tasks with batch mode..."
            )
            batch_metrics = self.batch_evaluate_complement(tasks_by_type["complement"])
            metrics.extend(batch_metrics)

        # Print metrics
        counter = Counter()
        empty_counter = Counter()
        success_counter = Counter()
        gt_counter = Counter()
        match_counter = Counter()
        budget_counter = Counter()
        recognition_counter = Counter()

        dataset_suffix = "test" if "_test" in args.ref_file else "train"
        reject_sample_file = args.hyp_file \
            .replace("rollout_", "reject_sample_") \
            .replace(".jsonl", f"_{dataset_suffix}.jsonl")
        with open(reject_sample_file, "w") as fout:
            for metric in metrics:
                question_type = metric["question_type"]
                question_id = metric["question_id"]

                is_success = 0
                if (
                    metric["n"] > 0
                    and (metric["gt"] + metric["match"]) == metric["n"]
                    and metric["n"] >= metric["n_gt"]
                    and metric.get("budget", 1) == 1
                ):
                    is_success = 1
                    fout.write(json.dumps(self.id2trej[question_id]) + "\n")
                    if metric["gt"] > 0:
                        gt_counter[question_type] += 1
                    if metric["match"] > 0:
                        match_counter[question_type] += 1
                if metric["n"] == 0:
                    empty_counter[question_type] += 1
                if "budget" in metric:
                    budget_counter[question_type] += metric["budget"]
                if "recognition" in metric:
                    recognition_counter[question_type] += metric["recognition"]

                success_counter[question_type] += is_success
                counter[question_type] += 1

        print(f"The evaluation results of {args.hyp_file} are as follows:")
        for question_type, total in counter.items():
            empty = empty_counter[question_type]
            success = success_counter[question_type]
            gt = gt_counter[question_type]
            match = match_counter[question_type]
            print("-" * 10 + question_type + "-" * 10)
            print(f"total: {total}, success rate: {success / total: .3f}, gt rate: {gt / total: .3f}, match rate: {match / total: .3f}, empty rate: {empty / total: .3f}")
            if question_type == "add_on_deals":
                print(f"budget fulfillment rate: {budget_counter[question_type] / total: .3f}")
            if question_type == "complement":
                print(f"recognition rate: {recognition_counter[question_type] / total: .3f}")

    def batch_evaluate_single_product(
        self, tasks: List[Tuple[ConversationDTO, List[Dict]]]
    ) -> List[Dict[str, Any]]:
        """
        批量评估多个 single_product 任务，最大化并行效率

        Args:
            tasks: (conversation, hypothesis) 元组列表

        Returns:
            metric 字典列表
        """
        metrics = []
        all_llm_requests = []
        task_request_mappings = []  # 记录每个任务的请求范围和产品映射

        # 第一遍：收集所有需要评估的产品和LLM请求
        for task_idx, (conversation, hypothesis) in enumerate(tasks):
            metric = {
                "question_type": conversation.question_type,
                "question_id": conversation.question_id,
                "n_gt": 1,
                "n": 0,
                "gt": 0,
                "match": 0,
            }

            # 提取答案
            hyp_answer = ""
            i = len(hypothesis) - 1
            while i >= 0:
                step = hypothesis[i]
                reward_model = step.get("reward_model", {})
                content = reward_model.get("ground_truth", "")

                if (
                    content
                    and "<answer>" in content
                    and "</answer>" in content
                    and content.index("<answer>") < content.index("</answer>")
                ):
                    hyp_answer = (
                        content.split("<answer>")[1].split("</answer>")[0].strip()
                    )
                    break

                i -= 1

            mathobj = re.search(r"@REC::([0-9,]+?)@", hyp_answer)
            if not mathobj:
                metrics.append(metric)
                task_request_mappings.append(None)
                continue

            product_ids = mathobj.group(1).split(",")
            if not product_ids or len(set(product_ids)) != len(product_ids):
                metrics.append(metric)
                task_request_mappings.append(None)
                continue

            # 加载产品
            docs = [self.searcher.doc(product_id) for product_id in product_ids]
            if any(not doc for doc in docs):
                metrics.append(metric)
                task_request_mappings.append(None)
                continue
            products = [json.loads(doc.raw())["product"] for doc in docs]

            metric["n"] = len(products)

            # 收集该任务的LLM请求
            product_request_mapping = []

            for product in products:
                if product["product_id"] == conversation.answer["product_id"]:
                    metric["gt"] += 1
                else:
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
                    wanted_features = ""
                    for feature in conversation.answer["wanted_features"]:
                        wanted_features += f"- {feature}\n"

                    # user query
                    user_query = conversation.question

                    prompt = single_product_evaluation_prompt.format(
                        user_query=user_query,
                        wanted_features=wanted_features,
                        recommended_product=recommended_product,
                    )
                    if self.args.debug:
                        print(f"Single product evaluation prompt:\n{prompt}", file=sys.stderr)
                    product_request_mapping.append(len(all_llm_requests))
                    all_llm_requests.append(
                        CompletionRequest(
                            messages=[{"role": "user", "content": prompt}],
                            model="gpt-5-2025-08-07-GlobalStandard",
                            extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                        )
                    )


            metrics.append(metric)
            task_request_mappings.append(product_request_mapping)

        # 第二遍：一次性批量调用所有LLM请求
        if all_llm_requests:
            print(f"Batch calling {len(all_llm_requests)} LLM requests...")
            llm_responses = self.client.batch_complete(all_llm_requests, verbose=True)

            # 第三遍：根据映射关系处理结果
            assert len(metrics) == len(task_request_mappings)
            for task_idx, task_mapping in enumerate(task_request_mappings):
                if task_mapping is None:
                    continue

                metric = metrics[task_idx]
                assert metric["n"] > 0

                for request_index in task_mapping:
                    response = llm_responses[request_index]

                    if response.success:
                        if "yes" == response.content.strip():
                            metric["match"] += 1
                    else:
                        print(response.error, file=sys.stderr)
        return metrics

    def batch_evaluate_add_on_deals(
        self, tasks: List[Tuple[ConversationDTO, List[Dict]]]
    ) -> List[Dict[str, Any]]:
        metrics = []
        all_llm_requests = []
        task_request_mappings = []  # 记录每个任务的请求范围和产品映射

        # 第一遍：收集所有需要评估的产品和LLM请求
        for task_idx, (conversation, hypothesis) in enumerate(tasks):
            metric = {
                "question_type": conversation.question_type,
                "question_id": conversation.question_id,
                "n_gt": conversation.answer["n"],
                "n": 0,
                "gt": 0,
                "match": 0,
                "budget": 0,
            }

            # 提取答案
            hyp_answer = ""
            i = len(hypothesis) - 1
            while i >= 0:
                step = hypothesis[i]
                reward_model = step.get("reward_model", {})
                content = reward_model.get("ground_truth", "")

                if (
                    content
                    and "<answer>" in content
                    and "</answer>" in content
                    and content.index("<answer>") < content.index("</answer>")
                ):
                    hyp_answer = (
                        content.split("<answer>")[1].split("</answer>")[0].strip()
                    )
                    break

                i -= 1

            mathobj = re.search(r"@REC::([0-9,]+?)@", hyp_answer)
            if not mathobj:
                metrics.append(metric)
                task_request_mappings.append(None)
                continue

            product_ids = mathobj.group(1).split(",")
            if not product_ids or len(set(product_ids)) != len(product_ids):
                metrics.append(metric)
                task_request_mappings.append(None)
                continue

            # 加载产品
            docs = [self.searcher.doc(product_id) for product_id in product_ids]
            if any(not doc for doc in docs):
                metrics.append(metric)
                task_request_mappings.append(None)
                continue
            products = [json.loads(doc.raw())["product"] for doc in docs]

            metric["n"] = len(products)

            if metric["n"] != metric["n_gt"]:
                metrics.append(metric)
                task_request_mappings.append(None)
                continue

            # voucher
            voucher_type = conversation.answer["voucher_type"]
            threshold = conversation.answer["voucher"]["threshold"]
            discount = conversation.answer["voucher"]["discount"]
            cap = conversation.answer["voucher"]["cap"]
            budget = conversation.answer["voucher"]["budget"]
            shop_ids = set(product["seller_id"] for product in products)

            total_price = sum(product["price"] for product in products)
            if voucher_type == "platform" and total_price >= threshold:
                price_after_voucher = max(
                    total_price - cap,
                    total_price * (1 - float(discount.strip("%")) / 100.0),
                )
            elif voucher_type == "shop" and len(shop_ids) == 1 and total_price >= threshold:
                price_after_voucher = max(
                    total_price - cap,
                    total_price * (1 - float(discount.strip("%")) / 100.0),
                )
            else:
                price_after_voucher = total_price

            metric["budget"] = 1 if price_after_voucher <= budget else 0

            # gt
            ref_product_ids = [p["product_id"] for p in conversation.answer["preferences"]]
            if sorted(ref_product_ids) == sorted(product_ids):
                metric["gt"] = metric["n"]
                metrics.append(metric)
                task_request_mappings.append(None)
                continue

            # 收集该任务的LLM请求
            # user query
            user_query = conversation.question

            # wanted features
            wanted_features = []
            for i, preference in enumerate(conversation.answer["preferences"]):
                f = f"The wanted features for product {i+1} in the user query are:\n"
                for feature in preference["wanted_features"]:
                    f += f"- {feature}\n"
                wanted_features.append(f.strip())

            wanted_features = "\n\n".join(wanted_features)

            # recommended products
            recommended_products = []
            for product in products:
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
                recommended_products.append(recommended_product)
            recommended_products = "\n\n".join(recommended_products)

            prompt = add_on_deals_evaluation_prompt.format(
                user_query=user_query,
                wanted_features=wanted_features,
                recommended_products=recommended_products,
            )
            if self.args.debug:
                print(f"Add on deals evaluation prompt:\n{prompt}", file=sys.stderr)

            metrics.append(metric)
            task_request_mappings.append(len(all_llm_requests))
            all_llm_requests.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-2025-08-07-GlobalStandard",
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                )
            )

        # 第二遍：一次性批量调用所有LLM请求
        if all_llm_requests:
            print(f"Batch calling {len(all_llm_requests)} LLM requests...")
            llm_responses = self.client.batch_complete(all_llm_requests, verbose=True)

            # 第三遍：根据映射关系处理结果
            for task_idx, request_idx in enumerate(task_request_mappings):
                if request_idx is None:
                    continue

                metric = metrics[task_idx]
                response = llm_responses[request_idx]

                if response.success:
                    if self.args.debug:
                        print(f"Add on deals evaluation response:\n{response.content}", file=sys.stderr)

                    if "yes" == response.content.strip():
                        metric["match"] = metric["n"]
                else:
                    print(response.error, file=sys.stderr)

        return metrics

    def batch_evaluate_repeat_purchase(
        self, tasks: List[Tuple[ConversationDTO, List[Dict]]]
    ) -> List[Dict[str, Any]]:
        metrics = []
        all_llm_requests = []
        task_request_mappings = []  # 记录每个任务的请求范围和产品映射

        # 第一遍：收集所有需要评估的产品和LLM请求
        for task_idx, (conversation, hypothesis) in enumerate(tasks):
            metric = {
                "question_type": conversation.question_type,
                "question_id": conversation.question_id,
                "n_gt": 0,
                "n": 0,
                "gt": 0,
                "match": 0,
            }

            # 提取答案
            hyp_answers = []
            i = 0
            while i < len(hypothesis):
                step = hypothesis[i]
                reward_model = step.get("reward_model", {})
                content = reward_model.get("ground_truth", "")

                if (
                    content
                    and "<answer>" in content
                    and "</answer>" in content
                    and content.index("<answer>") < content.index("</answer>")
                ):
                    hyp_answers.append(content.split("<answer>")[1].split("</answer>")[0].strip())

                i += 1
            hyp_answer = "\n\n".join(hyp_answers)

            # 收集该任务的LLM请求
            ref_product_name = conversation.answer["product_name"]
            ref_cycle = conversation.answer["cycle"]
            ref_should_repurchase = conversation.answer["should_repurchase"]
            reference = f"Product Name: {ref_product_name}\nRepurchase Cycle: {ref_cycle}\nShould Repurchase: {ref_should_repurchase}"

            prompt = repeat_purchase_evaluation_prompt.format(
                user_query=conversation.question,
                reference=reference,
                hypothesis_answer=hyp_answer,
            )
            if self.args.debug:
                print(f"Repeat purchase evaluation prompt:\n{prompt}", file=sys.stderr)

            metrics.append(metric)
            task_request_mappings.append(len(all_llm_requests))
            all_llm_requests.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-2025-08-07-GlobalStandard",
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                )
            )

        # 第二遍：一次性批量调用所有LLM请求
        if all_llm_requests:
            print(f"Batch calling {len(all_llm_requests)} LLM requests...")
            llm_responses = self.client.batch_complete(all_llm_requests, verbose=True)

            # 第三遍：根据映射关系处理结果
            for task_idx, request_idx in enumerate(task_request_mappings):
                if request_idx is None:
                    continue

                metric = metrics[task_idx]
                response = llm_responses[request_idx]

                if response.success:
                    if "yes" == response.content:
                        metric["n"] = 1
                        metric["match"] = 1
                else:
                    print(response.error, file=sys.stderr)

        return metrics

    def batch_evaluate_complement(
        self, tasks: List[Tuple[ConversationDTO, List[Dict]]]
    ) -> List[Dict[str, Any]]:
        metrics = []
        all_llm_requests = []
        all_recognition_llm_requests = []
        task_request_mappings = []  # 记录每个任务的请求范围和产品映射

        # 第一遍：收集所有需要评估的产品和LLM请求
        for task_idx, (conversation, hypothesis) in enumerate(tasks):
            metric = {
                "question_type": conversation.question_type,
                "question_id": conversation.question_id,
                "n_gt": 1,
                "n": 0,
                "gt": 0,
                "match": 0,
                "recognition": 0,
            }

            # 提取答案
            hyp_answers = []
            i = 0
            while i < len(hypothesis):
                step = hypothesis[i]
                reward_model = step.get("reward_model", {})
                content = reward_model.get("ground_truth", "")

                if (
                    content
                    and "<answer>" in content
                    and "</answer>" in content
                    and content.index("<answer>") < content.index("</answer>")
                ):
                    hyp_answers.append(content.split("<answer>")[1].split("</answer>")[0].strip())

                i += 1

            hyp_answer = "\n\n".join(hyp_answers)

            # 提取商品
            mathobj = re.search(r"@REC::([0-9,]+?)@", hyp_answer)
            if not mathobj:
                metrics.append(metric)
                task_request_mappings.append(None)
                continue

            product_ids = mathobj.group(1).split(",")
            if not product_ids or len(set(product_ids)) != len(product_ids):
                metrics.append(metric)
                task_request_mappings.append(None)
                continue

            # 加载产品
            docs = [self.searcher.doc(product_id) for product_id in product_ids]
            if any(not doc for doc in docs):
                metrics.append(metric)
                task_request_mappings.append(None)
                continue
            products = [json.loads(doc.raw())["product"] for doc in docs]

            metric["n"] = len(products)

            # 收集该任务的LLM请求
            purchased_product = "\n".join([f"{i+1}. {name}" for i, name in enumerate(conversation.answer["product_names"])])
            recommended_products = "\n".join([f"{i+1}. {p['product_name']}" for i, p in enumerate(products)])

            prompt = complement_evaluation_prompt.format(
                purchased_product=purchased_product,
                recommended_products=recommended_products,
            )
            if self.args.debug:
                print(f"Complement evaluation prompt:\n{prompt}", file=sys.stderr)

            metrics.append(metric)
            task_request_mappings.append(len(all_llm_requests))
            all_llm_requests.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-2025-08-07-GlobalStandard",
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                )
            )

            # 收集识别任务的LLM请求
            prompt = complement_recognition_prompt.format(
                reference_purchased_product=purchased_product,
                hypothesis_answer=hyp_answer,
            )
            if self.args.debug:
                print(f"Complement recognition prompt:\n{prompt}", file=sys.stderr)
            all_recognition_llm_requests.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-2025-08-07-GlobalStandard",
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                )
            )

        # 第二遍：一次性批量调用所有LLM请求
        if all_llm_requests and all_recognition_llm_requests:
            print(f"Batch calling {len(all_llm_requests)} LLM requests...")
            llm_responses = self.client.batch_complete(all_llm_requests, verbose=True)
            print(f"Batch calling {len(all_recognition_llm_requests)} recognition LLM requests...")
            recognition_llm_responses = self.client.batch_complete(all_recognition_llm_requests, verbose=True)

            # 第三遍：根据映射关系处理结果
            for task_idx, request_idx in enumerate(task_request_mappings):
                if request_idx is None:
                    continue

                metric = metrics[task_idx]
                response = llm_responses[request_idx]
                recognition_response = recognition_llm_responses[request_idx]

                if response.success:
                    if "yes" == response.content:
                        metric["match"] = metric["n"]
                else:
                    print(response.error, file=sys.stderr)

                if recognition_response.success:
                    if "yes" == recognition_response.content:
                        metric["recognition"] = 1
                else:
                    print(recognition_response.error, file=sys.stderr)

        return metrics

    def evaluate_preference(self):
        # Load searcher
        self.searcher = LuceneSearcher(self.args.index_dir)

        # Load client
        self.client = ParallelOpenAICompletion(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=self.args.max_workers,
            max_retries=3,
            retry_delay=1,
        )

        # Load id2conv
        with open(self.args.ref_file, "r") as fin:
            for line in tqdm(fin, desc="Loading id2conv: "):
                item = json.loads(line.strip())
                conversation = ConversationDTO(**item)

                self.id2conv[conversation.question_id] = conversation

        # Load conversation
        tasks_by_type = {
            "single_product": [],
            "add_on_deals": [],
            "repeat_purchase": [],
            "complement": [],
        }

        with open(self.args.hyp_file, "r") as fin:
            for line in fin:
                item = json.loads(line.strip())

                if not item:
                    continue

                question_id = item[0].get("extra_info", {}).get("question_id")
                if not question_id:
                    continue

                if question_id not in self.id2conv:
                    continue
                conversation = self.id2conv[question_id]

                if not conversation.answer:
                    print(
                        f"No answer for conversation: {question_id}",
                        file=sys.stderr,
                    )
                    continue

                question_type = item[0].get("extra_info", {}).get("question_type")

                if question_type in tasks_by_type:
                    tasks_by_type[question_type].append((conversation, item))
                    self.id2trej[question_id] = item
                else:
                    print(f"Unknown question type: {question_type}", file=sys.stderr)

        # 按类型批量评估
        metrics = []        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--ref_file", type=str, required=True)
    parser.add_argument("--hyp_file", type=str, required=True)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--mode", type=str, default="evaluation", choices=["evaluation", "preference"])
    args = parser.parse_args()

    evaluator = Evaluator(args=args)
    if args.mode == "evaluation":
        evaluator.evaluate()
    elif args.mode == "preference":
        evaluator.evaluate_preference()
