import os
import time
import logging
import re
import json
import asyncio
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from pyserini.search.lucene import LuceneSearcher
import sys
from pathlib import Path
import regex




from reward_prompt import (
    single_product_stage_1_evaluation_prompt,
    single_product_stage_2_evaluation_prompt,
    add_on_deals_stage_1_evaluation_prompt,
    add_on_deals_stage_2_evaluation_prompt,
    add_on_deals_evaluation_prompt,
    single_product_evaluation_prompt
)

MAX_RETRIES = 10

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()

# 全局 searcher 和 index_dir
_searcher = None
_index_dir = None

# 全局线程池（用于在协程中运行同步的 ask_llm）
_executor = ThreadPoolExecutor(max_workers=4)

def get_index_dir():
    """动态获取索引目录路径"""
    project_root = Path(__file__).resolve().parents[1] 
    return str(project_root / "data" / "product_indexes")

async def ask_llm_async(
    messages: list[dict[str, str]],
    model_config: dict,
    base_url: str,
    api_key: str,
) -> tuple[str, str]:
    """异步版本 LLM 调用"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: ask_llm(messages, model_config, base_url, api_key)
    )

async def evaluate_single_product_stage_1_async(
    question: str,
    answer: str,
    ground_truth_dict: Dict[str, Any],
    base_url: str,
    api_key: str,
) -> float:
    
    feature_num = 0
    wanted_features = ""
    for feature in ground_truth_dict.get("wanted_features", []):
        wanted_features += f"- {feature}\n"
        feature_num += 1

    product_name = ground_truth_dict.get("product_name", [])


    prompt = single_product_stage_1_evaluation_prompt.format(
        question=question,
        wanted_features=wanted_features,
        product_name=product_name,
        answer=answer
    )
    
    try:
        _, content = await ask_llm_async(
            messages=[{"role": "user", "content": prompt}],
            model_config={
                "model": "gpt-5.1-2025-11-13-GlobalStandard",
                "stream": False,
            },
            base_url=base_url,
            api_key=api_key,
        )
        scores_dict = parse_json_response(content)
        print("任务 1 stage 1", scores_dict)

        query_relevance = float(scores_dict.get("query_relevance", 0))
        preference_match_count = float(scores_dict.get("preference_match_count", 0))
        
        orm_score = min(1.0,(query_relevance + preference_match_count) / (1.0 + float(feature_num)))

        return orm_score

    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")
        return 0.0

async def evaluate_single_product_stage_2_async(
    question: str,
    answer: str,
    ground_truth_dict: Dict[str, Any],
    base_url: str,
    api_key: str,
) -> tuple[float, float]:
    
    product_format_score = 0.0

    mathobj = re.search(r"@REC::([0-9,]+?)@", answer)
    if not mathobj:
        logger.warning("Could not extract product ID from answer")
        return product_format_score, 0.0
    else:
        product_format_score = 1.0
    
    product_ids = mathobj.group(1).split(",")

    gt_product_id = ground_truth_dict.get("product_id", "")

    if len(product_ids) == 1 and product_ids[0] == gt_product_id:
        logger.info("Perfect match - product ID matches ground truth")
        return product_format_score, 1.0
    
    searcher = get_searcher()
    
    try:
        docs = [searcher.doc(pid) for pid in product_ids]
        if any(not doc for doc in docs):
            logger.warning("Could not load product details from searcher")
            return product_format_score, 0.0
        products = [json.loads(doc.raw())["product"] for doc in docs]

    except Exception as e:
        logger.error(f"Failed to load product details: {e}")
        return product_format_score, 0.0

    is_product = 1.0

    feature_num = 0
    wanted_features = ""
    for feature in ground_truth_dict.get("wanted_features", []):
        wanted_features += f"- {feature}\n"
        feature_num += 1

    products_info = ""
    for i, product in enumerate(products):
        product_name = product.get("product_name", "")
        attributes = product.get("attributes", {})
        options = product.get("options", [])
        
        kv = dict()
        for k, vs in attributes.items():
            for v in vs:
                kv[k] = v
        
        if options:
            for option in options:
                for k, vs in option.items():
                    for v in vs:
                        kv[k] = v
        
        attributes_str = "\n".join([f"{k}: {v}" for k, v in kv.items()])
        products_info += f"\n{i+1}. Product Name: {product_name}\nAttributes:\n{attributes_str}"
    
    prompt = single_product_stage_2_evaluation_prompt.format(
        question=question,
        wanted_features=wanted_features,
        products_info=products_info
    )
    
    try:
        _, content = await ask_llm_async(
            messages=[{"role": "user", "content": prompt}],
            model_config={
                "model": "gpt-5.1-2025-11-13-GlobalStandard",
                "stream": False,
            },
            base_url=base_url,
            api_key=api_key,
        )
        
        scores_dict = parse_json_response(content)
        print("任务 1 stage 2", scores_dict)

        query_relevance = float(scores_dict.get("query_relevance", 0))
        preference_match_count = float(scores_dict.get("preference_match_count", 0))

        orm_score = min(1.0,(is_product + query_relevance + preference_match_count) / (1.0 + 1.0 + float(feature_num)))

        return product_format_score, orm_score

    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")
        return product_format_score, 0.0

async def evaluate_single_product_offline_metric_async(
    question: str,
    answer: str,
    ground_truth_dict: Dict[str, Any],
    base_url: str,
    api_key: str,
) -> tuple[float, float, float, float]:
    """
    Evaluates a single product recommendation.

    Returns:
        (empty_rate, gt_rate, match_rate, success_rate)
        each is 0.0 or 1.0 for a single sample
    """
    mathobj = re.search(r"@REC::([0-9,]+?)@", answer)
    if not mathobj:
        logger.warning("Could not extract product ID from answer")
        return 1.0, 0.0, 0.0, 0.0
    
    product_ids_str = mathobj.group(1).strip()
    if not product_ids_str:
        logger.warning("Empty product ID list after extraction")
        return 1.0, 0.0, 0.0, 0.0
    
    product_ids = [pid.strip() for pid in product_ids_str.split(",") if pid.strip()]
    
    searcher = get_searcher()
    try:
        docs = [searcher.doc(pid) for pid in product_ids]
        if any(doc is None for doc in docs):
            failed_pids = [pid for pid, doc in zip(product_ids, docs) if doc is None]
            logger.warning(f"Failed to retrieve products for IDs: {failed_pids}")
            return 1.0, 0.0, 0.0, 0.0
        products = [json.loads(doc.raw())["product"] for doc in docs]
    except Exception as e:
        logger.error(f"Failed to load product details: {e}")
        return 1.0, 0.0, 0.0, 0.0
    
    gt_product_id = ground_truth_dict.get("product_id", "")
    gt_rate = 0.0
    
    for product in products:
        if product["product_id"] == gt_product_id:
            gt_rate = 1.0
            logger.info(f"GT match found: {gt_product_id}")
            return 0.0, 1.0, 0.0, 1.0
    
    wanted_features_gt = ground_truth_dict.get("wanted_features", [])
    match_rate = 0.0
    
    for product in products:
        if product["product_id"] == gt_product_id:
            continue
        
        product_name = product.get("product_name", "")
        attributes = product.get("attributes", {})
        options = product.get("options", [])
        
        attributes_str = ""
        if attributes:
            attributes_str = "; ".join(
                [f"{k} = {', '.join(vs)}" 
                 for k, vs in sorted(attributes.items(), key=lambda x: x[0])]
            )
        
        options_str = ""
        if options:
            for option in options:
                if option:
                    options_str += "- " + "; ".join(
                        [f"{k} = {', '.join(vs)}" 
                         for k, vs in sorted(option.items(), key=lambda x: x[0])]
                    ) + "\n"
        
        recommended_product = f"Product Name: {product_name}\nAttributes: {attributes_str}\nOptions:\n{options_str}".strip()
        
        wanted_features_text = "\n".join([f"- {feature}" for feature in wanted_features_gt])
        
        prompt = single_product_evaluation_prompt.format(
            user_query=question,
            wanted_features=wanted_features_text,
            recommended_product=recommended_product,
        )
        
        try:
            _, response = await ask_llm_async(
                messages=[{"role": "user", "content": prompt}],
                model_config={
                    "model": "gpt-5.1-2025-11-13-GlobalStandard",
                    "stream": False,
                },
                base_url=base_url,
                api_key=api_key,
            )
            
            cleaned = response.strip().lower()
            if "yes" in cleaned:
                match_rate = 1.0
                logger.info(f"LLM match found for product: {product_name}")
                break
        except Exception as e:
            logger.warning(f"LLM evaluation failed for product {product_name}: {e}")
            continue
    
    success_rate = 1.0 if (gt_rate > 0.0 or match_rate > 0.0) else 0.0
    
    return 0.0, gt_rate, match_rate, success_rate

async def evaluate_add_on_deals_stage_1_async(
    question: str,
    answer: str,
    ground_truth_dict: Dict[str, Any],
    base_url: str,
    api_key: str,
) -> float:
    
    reference_products = []
    for i, pref in enumerate(ground_truth_dict.get("preferences", [])):
        product_name = pref.get("product_name", "")
        wanted_features = "\n".join(pref.get("wanted_features", []))
        reference_products.append(
            f"{i+1}. Product Name: {product_name}\nWanted Features:\n{wanted_features}"
        )
    reference_product_bundle = "\n\n".join(reference_products)
    
    gt_n = ground_truth_dict.get("n", 0)

    feature_num = 0
    for i, preference in enumerate(ground_truth_dict.get("preferences", [])):
        for feature in preference.get("wanted_features", []):
            feature_num += 1

    
    prompt = add_on_deals_stage_1_evaluation_prompt.format(
        question=question,
        reference_product_bundle=reference_product_bundle,
        answer=answer
    )
    
    try:
        _, content = await ask_llm_async(
            messages=[{"role": "user", "content": prompt}],
            model_config={
                "model": "gpt-5.1-2025-11-13-GlobalStandard",
                "stream": False,
            },
            base_url=base_url,
            api_key=api_key,
        )
        
        scores_dict = parse_json_response(content)
        print("任务 2 stage 1", scores_dict)

        query_relevance = float(scores_dict.get("query_relevance", 0))
        product_count = float(scores_dict.get("product_count", 0))
        preference_match_count = float(scores_dict.get("preference_match_count", 0))
        
        orm_score = min(1.0,(query_relevance + product_count + preference_match_count) / (1.0 + float(gt_n) + float(feature_num)))

        return orm_score

    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")
        return 0.0

async def evaluate_add_on_deals_stage_2_async(
    question: str,
    answer: str,
    ground_truth_dict: Dict[str, Any],
    base_url: str,
    api_key: str,
) -> tuple[float, float]:

    product_format_score = 0.0

    mathobj = re.search(r"@REC::([0-9,]+?)@", answer)
    if not mathobj:
        logger.warning("Could not extract product IDs from answer")
        return product_format_score, 0.0
    else:
        product_format_score = 1.0
    
    product_ids = mathobj.group(1).split(",")

    gt_n = ground_truth_dict.get("n", 0)

    if len(product_ids) == gt_n:
        number_check = 1.0
    else:
        number_check = 0.0

    searcher = get_searcher()
    
    try:
        docs = [searcher.doc(pid) for pid in product_ids]
        if any(not doc for doc in docs):
            logger.warning("Could not load product details")
            return product_format_score, 0.0
        
        products = [json.loads(doc.raw())["product"] for doc in docs]
    except Exception as e:
        logger.error(f"Failed to load products: {e}")
        return product_format_score, 0.0

    budget_satisfied = _check_budget(products, ground_truth_dict)

    wanted_features_list = []
    feature_num = 0
    for i, preference in enumerate(ground_truth_dict.get("preferences", [])):
        f = f"The wanted features for product {i+1} in the user query are:\n"
        for feature in preference.get("wanted_features", []):
            f += f"- {feature}\n"
            feature_num += 1
        wanted_features_list.append(f.strip())
    
    wanted_features_text = "\n\n".join(wanted_features_list)

    # 检查是否完全匹配参考答案
    ref_product_ids = [p["product_id"] for p in ground_truth_dict.get("preferences", [])]
    if sorted(ref_product_ids) == sorted(product_ids):
        # 检查预算
        if budget_satisfied:
            return product_format_score, 1.0
        else:
            orm_score = float(1 - 1 / (2 + feature_num + gt_n))
            return product_format_score, orm_score

    # 构建推荐商品列表
    recommended_products_list = []
    for product in products:
        product_name = product.get("product_name", "")
        attributes = product.get("attributes", {})
        options = product.get("options", [])
        
        attributes_str = ""
        if attributes:
            attributes_str = "; ".join(
                [f"{k} = {', '.join(vs)}" 
                 for k, vs in sorted(attributes.items(), key=lambda x: x[0])]
            )
        
        options_str = ""
        if options:
            for option in options:
                if option:
                    options_str += "- " + "; ".join(
                        [f"{k} = {', '.join(vs)}" 
                         for k, vs in sorted(option.items(), key=lambda x: x[0])]
                    ) + "\n"
        
        recommended_product = f"Product Name: {product_name}\nAttributes: {attributes_str}\nOptions:\n{options_str}".strip()
        recommended_products_list.append(recommended_product)
    
    recommended_products_text = "\n\n".join(recommended_products_list)
    
    prompt = add_on_deals_stage_2_evaluation_prompt.format(
        user_query=question,
        wanted_features=wanted_features_text,
        recommended_products=recommended_products_text,
    )
    
    try:
        _, content = await ask_llm_async(
            messages=[{"role": "user", "content": prompt}],
            model_config={
                "model": "gpt-5.1-2025-11-13-GlobalStandard",
                "stream": False,
            },
            base_url=base_url,
            api_key=api_key,
        )
        
        scores_dict = parse_json_response(content)
        print("任务 2 stage 2", scores_dict)

        query_relevance_count = float(scores_dict.get("query_relevance_count", 0))
        feature_match_count = float(scores_dict.get("feature_match_count", 0))
        

        orm_score = min(1.0, (float(number_check) + float(budget_satisfied) + query_relevance_count + feature_match_count) / (2.0 + gt_n + feature_num))

        return product_format_score, orm_score

    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")
        return product_format_score, 0.0

async def evaluate_add_on_deals_offline_metric_async(
    question: str,
    answer: str,
    ground_truth_dict: Dict[str, Any],
    base_url: str,
    api_key: str,
) -> tuple[float, float, float, float, float]:
    """
    Evaluates an add-on deals recommendation (multiple products).

    Returns:
        (empty_rate, budget_rate, gt_rate, match_rate, success_rate)
        each is 0.0 or 1.0 for a single sample
    """
    
    # Step 1: Extract product IDs
    mathobj = re.search(r"@REC::([0-9,]+?)@", answer)
    if not mathobj:
        logger.warning("Could not extract product IDs from answer")
        return 1.0, 0.0, 0.0, 0.0, 0.0
    
    product_ids_str = mathobj.group(1).strip()
    if not product_ids_str:
        logger.warning("Empty product ID list after extraction")
        return 1.0, 0.0, 0.0, 0.0, 0.0
    
    product_ids = [pid.strip() for pid in product_ids_str.split(",") if pid.strip()]
    
    # Check for duplicates
    if len(set(product_ids)) != len(product_ids):
        logger.warning("Duplicate product IDs found")
        return 1.0, 0.0, 0.0, 0.0, 0.0
    
    # Step 2: Load products
    searcher = get_searcher()
    try:
        docs = [searcher.doc(pid) for pid in product_ids]
        if any(doc is None for doc in docs):
            failed_pids = [pid for pid, doc in zip(product_ids, docs) if doc is None]
            logger.warning(f"Failed to retrieve products for IDs: {failed_pids}")
            return 1.0, 0.0, 0.0, 0.0, 0.0
        products = [json.loads(doc.raw())["product"] for doc in docs]
    except Exception as e:
        logger.error(f"Failed to load product details: {e}")
        return 1.0, 0.0, 0.0, 0.0, 0.0
    
    # Step 3: Check budget constraint
    if _check_budget(products, ground_truth_dict):
        budget_rate = 1.0
    else:
        budget_rate = 0.0
    
    # Step 4: Check GT match (all products must match)
    gt_product_ids = [p["product_id"] for p in ground_truth_dict.get("preferences", [])]
    gt_rate = 0.0
    
    if sorted(gt_product_ids) == sorted(product_ids):
        gt_rate = 1.0
        logger.info("GT match: all products match ground truth")
        # GT match found, return success if budget also ok
        success_rate = 1.0 if budget_rate == 1.0 else 0.0
        return 0.0, budget_rate, 1.0, 0.0, success_rate
    
    # Step 5: If no GT match, evaluate all products with LLM
    match_rate = 0.0
    
    # Build wanted features for all products
    wanted_features_list = []
    for i, preference in enumerate(ground_truth_dict.get("preferences", [])):
        f = f"The wanted features for product {i+1} in the user query are:\n"
        for feature in preference.get("wanted_features", []):
            f += f"- {feature}\n"
        wanted_features_list.append(f.strip())
    
    wanted_features_text = "\n\n".join(wanted_features_list)
    
    # Build recommended products info
    recommended_products_list = []
    for product in products:
        product_name = product.get("product_name", "")
        attributes = product.get("attributes", {})
        options = product.get("options", [])
        
        attributes_str = ""
        if attributes:
            attributes_str = "; ".join(
                [f"{k} = {', '.join(vs)}" 
                 for k, vs in sorted(attributes.items(), key=lambda x: x[0])]
            )
        
        options_str = ""
        if options:
            for option in options:
                if option:
                    options_str += "- " + "; ".join(
                        [f"{k} = {', '.join(vs)}" 
                         for k, vs in sorted(option.items(), key=lambda x: x[0])]
                    ) + "\n"
        
        recommended_product = f"Product Name: {product_name}\nAttributes: {attributes_str}\nOptions:\n{options_str}".strip()
        recommended_products_list.append(recommended_product)
    
    recommended_products_text = "\n\n".join(recommended_products_list)
    
    # Call LLM to evaluate all products as a bundle
    prompt = add_on_deals_evaluation_prompt.format(
        user_query=question,
        wanted_features=wanted_features_text,
        recommended_products=recommended_products_text,
    )
    
    try:
        _, response = await ask_llm_async(
            messages=[{"role": "user", "content": prompt}],
            model_config={
                "model": "gpt-5.1-2025-11-13-GlobalStandard",
                "stream": False,
            },
            base_url=base_url,
            api_key=api_key,
        )
        
        cleaned = response.strip().lower()
        if "yes" in cleaned:
            match_rate = 1.0
            logger.info("LLM approved all recommended products")
    except Exception as e:
        logger.warning(f"LLM evaluation failed: {e}")
    
    # Step 6: Determine final success
    # Success = (GT or match approved) AND budget satisfied
    success_rate = 1.0 if (gt_rate > 0.0 or match_rate > 0.0) and budget_rate > 0.0 else 0.0
    
    return 0.0, budget_rate, gt_rate, match_rate, success_rate


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any]
) -> Dict[str, float]:
    
    base_url = os.getenv("OPENAI_BASE_URL","")
    api_key = os.getenv("OPENAI_API_KEY","")
    
    question = extra_info.get("question", "")
    question_type = extra_info.get("question_type", "")
    ability = extra_info.get("ability", "")
    tool_rewards = extra_info.get("tool_rewards", [])
    assistant_turns = extra_info.get("assistant_turns", "")

    solution_str = "<|im_start|>assistant <think>" + solution_str

    tool_call_score, think_score = compute_tool_call_and_think_score(solution_str, assistant_turns)

    if not tool_rewards:
        avg_tool_score = 0.0
    else:
        avg_tool_score = sum(tool_rewards) / len(tool_rewards)

    index_dir = get_index_dir()
    init_searcher(index_dir)
    
    gt_dict = json.loads(ground_truth)
    
    answer = extract_solution(solution_str)
    answer_format_score = 0.0
    product_format_score = 0.0
    orm_score = 0.0

    gt_rate1 = 0.0
    gt_rate2 = 0.0
    success_rate1 = 0.0
    success_rate2 = 0.0
    match_rate1 = 0.0
    match_rate2 = 0.0
    empty_rate1 = 0.0
    empty_rate2 = 0.0
    budget_rate = 0.0

    if not answer:
        answer_format_score = 0.0
        if ability == "stage_1":
            score = avg_tool_score + orm_score + (answer_format_score + think_score + tool_call_score ) / 3
        else:
            score =  avg_tool_score + orm_score + (answer_format_score + think_score + tool_call_score + product_format_score) / 4
        _log_evaluation_error(
            ground_truth, 
            solution_str, 
            f"score={score}, product_format_score={product_format_score}, avg_tool_score={avg_tool_score}, orm_score={orm_score}, answer_format_score={answer_format_score}, think_score={think_score}, tool_call_score={tool_call_score}, gt_rate1={gt_rate1}, success_rate1={success_rate1}, match_rate1={match_rate1}, empty_rate1={empty_rate1}, gt_rate2={gt_rate2}, success_rate2={success_rate2}, match_rate2={match_rate2}, empty_rate2={empty_rate2}, budget_rate={budget_rate}",
            "no answer extracted", 
            score
        )
        return {
            'score': float(score), 
            'avg_tool_score': float(avg_tool_score), 
            'orm_score': float(orm_score),
            'product_format_score': float(product_format_score * 2),
            'answer_format_score': float(answer_format_score),
            'think_score': float(think_score),
            'tool_call_score': float(tool_call_score),
            'gt_rate1': float(gt_rate1 * 4), 
            'success_rate1': float(success_rate1 * 4), 
            'match_rate1': float(match_rate1 * 4), 
            'empty_rate1': float(empty_rate1 * 4),
            'gt_rate2': float(gt_rate2 * 4), 
            'success_rate2': float(success_rate2 * 4), 
            'match_rate2': float(match_rate2 * 4), 
            'empty_rate2': float(empty_rate2 * 4),
            'budget_rate': float(budget_rate * 4),
        }   
    else:
        answer_format_score = 1.0    
        try:
            if ability == "stage_1":
                if question_type == "single_product":
                    orm_score = await evaluate_single_product_stage_1_async(question, answer, gt_dict, base_url, api_key)
                else:
                    orm_score = await evaluate_add_on_deals_stage_1_async(question, answer, gt_dict, base_url, api_key)
                score = avg_tool_score + orm_score + (answer_format_score + think_score + tool_call_score ) / 3
            else:
                # ⭐ Stage 2: 并行调用两个异步函数
                if question_type == "single_product":
                    # 并行执行 stage_2 评估和离线指标评估
                    (product_format_score, orm_score), (empty_rate1, gt_rate1, match_rate1, success_rate1) = \
                        await asyncio.gather(
                            evaluate_single_product_stage_2_async(question, answer, gt_dict, base_url, api_key),
                            evaluate_single_product_offline_metric_async(question, answer, gt_dict, base_url, api_key)
                        )
                else:
                    # 并行执行 stage_2 评估和离线指标评估
                    (product_format_score, orm_score), (empty_rate2, budget_rate, gt_rate2, match_rate2, success_rate2) = \
                        await asyncio.gather(
                            evaluate_add_on_deals_stage_2_async(question, answer, gt_dict, base_url, api_key),
                            evaluate_add_on_deals_offline_metric_async(question, answer, gt_dict, base_url, api_key)
                        )
                score = avg_tool_score + orm_score + (answer_format_score + think_score + tool_call_score + product_format_score) / 4

            
            
            _log_evaluation_success(
                ground_truth, 
                solution_str, 
                f"score={score}, product_format_score={product_format_score}, avg_tool_score={avg_tool_score}, orm_score={orm_score}, answer_format_score={answer_format_score}, think_score={think_score}, tool_call_score={tool_call_score}, gt_rate1={gt_rate1}, success_rate1={success_rate1}, match_rate1={match_rate1}, empty_rate1={empty_rate1}, gt_rate2={gt_rate2}, success_rate2={success_rate2}, match_rate2={match_rate2}, empty_rate2={empty_rate2}, budget_rate={budget_rate}",
                score
            )
            
            return {
                'score': float(score), 
                'avg_tool_score': float(avg_tool_score), 
                'orm_score': float(orm_score),
                'product_format_score': float(product_format_score * 2),
                'answer_format_score': float(answer_format_score),
                'think_score': float(think_score),
                'tool_call_score': float(tool_call_score),
                'gt_rate1': float(gt_rate1 * 4), 
                'success_rate1': float(success_rate1 * 4), 
                'match_rate1': float(match_rate1 * 4), 
                'empty_rate1': float(empty_rate1 * 4),
                'gt_rate2': float(gt_rate2 * 4), 
                'success_rate2': float(success_rate2 * 4), 
                'match_rate2': float(match_rate2 * 4), 
                'empty_rate2': float(empty_rate2 * 4),
                'budget_rate': float(budget_rate * 4),
            }
            
        except Exception as e:
            logger.error(f"Exception occurred during score computation: {type(e).__name__}: {str(e)}")
            log_error("exception", question, exception=type(e).__name__, error_msg=str(e))
            if ability == "stage_1":
                score = avg_tool_score + orm_score + (answer_format_score + think_score + tool_call_score ) / 3
            else:
                score = avg_tool_score + orm_score + (answer_format_score + think_score + tool_call_score + product_format_score) / 4

            _log_evaluation_error(
                ground_truth, 
                solution_str, 
                f"score={score}, product_format_score={product_format_score}, avg_tool_score={avg_tool_score}, orm_score={orm_score}, answer_format_score={answer_format_score}, think_score={think_score}, tool_call_score={tool_call_score}, gt_rate1={gt_rate1}, success_rate1={success_rate1}, match_rate1={match_rate1}, empty_rate1={empty_rate1}, gt_rate2={gt_rate2}, success_rate2={success_rate2}, match_rate2={match_rate2}, empty_rate2={empty_rate2}, budget_rate={budget_rate}",
                f"{type(e).__name__}: {str(e)}", 
                score
            )
            return {
                'score': float(score), 
                'avg_tool_score': float(avg_tool_score), 
                'orm_score': float(orm_score),
                'product_format_score': float(product_format_score * 2),
                'answer_format_score': float(answer_format_score),
                'think_score': float(think_score),
                'tool_call_score': float(tool_call_score),
                'gt_rate1': float(gt_rate1 * 4), 
                'success_rate1': float(success_rate1 * 4), 
                'match_rate1': float(match_rate1 * 4), 
                'empty_rate1': float(empty_rate1 * 4),
                'gt_rate2': float(gt_rate2 * 4), 
                'success_rate2': float(success_rate2 * 4), 
                'match_rate2': float(match_rate2 * 4), 
                'empty_rate2': float(empty_rate2 * 4),
                'budget_rate': float(budget_rate * 4),
            }

# 这些函数保持原样，不用改成异步版本
# 因为它们不直接调用 LLM，只做数据处理

def extract_solution(solution_str: str) -> Optional[str]:
    """提取answer标签中的答案"""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if len(matches) < 1:
        return None
    
    return matches[-1].group(1).strip()



def compute_tool_call_and_think_score(solution_str: str, assistant_turns: int) -> tuple[float, float]:
    if assistant_turns <= 1:
        return 0.0, 0.0  # 避免除以零

    start_tag = "<|im_start|>assistant"
    end_tag = "<|im_end|>"
    tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)
    think_start = "<think>"
    think_end = "</think>"
    
    correct_tool_call_turns = 0
    correct_think_turns = 0
    i = 0
    
    # 逐轮处理
    while i < len(solution_str):
        # 查找当前轮的 assistant 开始
        assistant_start = solution_str.find(start_tag, i)
        if assistant_start == -1:
            break
        
        # 查找当前轮的结束标签
        content_start = assistant_start + len(start_tag)
        im_end_pos = solution_str.find(end_tag, content_start)
        if im_end_pos == -1:
            break
        
        # 提取当前轮的内容
        turn_content = solution_str[content_start:im_end_pos]
        
        # ========== 检查 tool_call 格式 ==========
        # 使用正则表达式找到所有 tool_call 块
        tool_call_blocks = tool_call_regex.findall(turn_content)
        
        # tool_call 检查条件：
        # 支持两种格式：
        # 1. 多个 <tool_call>...</tool_call> 块，每块一个 JSON
        # 2. 一个 <tool_call>...</tool_call> 块，块内多行 JSON
        if tool_call_blocks:
            all_valid_json = True
            for block in tool_call_blocks:
                # 按行处理，支持多行 JSON 对象
                lines = block.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except (json.JSONDecodeError, Exception):
                        all_valid_json = False
                        break
                
                if not all_valid_json:
                    break
            
            if all_valid_json:
                correct_tool_call_turns += 1
        
        # ========== 检查 think 格式 ==========
        think_start_pos = turn_content.find(think_start)
        think_end_pos = turn_content.find(think_end)
        
        # think 检查条件：
        # 1. 恰好有一对 think 标签
        # 2. <think> 在 </think> 之前
        # 3. 没有多余的 think 标签
        if (think_start_pos != -1 and 
            think_end_pos != -1 and 
            think_start_pos < think_end_pos and
            turn_content.find(think_start, think_start_pos + 1) == -1 and
            turn_content.find(think_end, think_end_pos + 1) == -1):
            correct_think_turns += 1
        
        # 移动到下一轮
        i = im_end_pos + len(end_tag)
    
    # 返回 (tool_call_score, think_score)
    # tool_call_score: correct_turns / (assistant_turns - 1)
    # think_score: correct_turns / assistant_turns
    tool_call_score = min(1.0, float(correct_tool_call_turns / (assistant_turns - 1)))
    think_score = float(correct_think_turns / assistant_turns)
    
    return tool_call_score, think_score




def log_error(error_type: str, question: str, **kwargs):
    """记录错误日志，保持不变"""
    from datetime import datetime
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "question": question,
        **kwargs
    }
    with open("error_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")


def _log_evaluation_success(ground_truth, solution_str, response, score):
    """记录评估成功，保持不变"""
    output_line = f"response_text: {response} | ground_truth: {ground_truth[:10]} | solution_str: {solution_str[:10]} | score: {score}\n"
    with open("evaluation_results.log", "a", encoding="utf-8") as f:
        f.write(output_line)
        f.flush()


def _log_evaluation_error(ground_truth, solution_str, response, error_msg, score):
    """记录评估错误，保持不变"""
    output_line = f"response_text: {response} | ground_truth: {ground_truth[:10]} | solution_str: {solution_str[:10]} | error: {error_msg} | score: {score}\n"
    with open("evaluation_results.log", "a", encoding="utf-8") as f:
        f.write(output_line)
        f.flush()


def parse_json_response(response: str) -> Dict[str, Any]:
    """解析JSON响应，保持不变"""
    response_cleaned = response.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(response_cleaned)
    except Exception as e:
        print(f"JSON parsing error in loads: {e}\nThe response is: {response}")

    try:
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


def _check_budget(products: list, ground_truth_dict: Dict[str, Any]) -> bool:
    """检查预算是否满足，保持不变"""
    voucher_info = ground_truth_dict.get("voucher", {})
    voucher_type = ground_truth_dict.get("voucher_type", "")
    
    threshold = float(voucher_info.get("threshold", 0))
    discount = voucher_info.get("discount", "0%")
    cap = float(voucher_info.get("cap", 0))
    budget = float(voucher_info.get("budget", float('inf')))
    
    total_price = sum(float(product.get("price", 0)) for product in products)
    
    # 检查是否满足阈值
    if voucher_type == "platform" and total_price >= threshold:
        discount_rate = float(discount.strip("%")) / 100.0
        price_after_voucher = max(
            total_price - cap,
            total_price * (1 - discount_rate),
        )
    elif voucher_type == "shop" and total_price >= threshold:
        discount_rate = float(discount.strip("%")) / 100.0
        price_after_voucher = max(
            total_price - cap,
            total_price * (1 - discount_rate),
        )
    else:
        price_after_voucher = total_price
    
    return price_after_voucher <= budget

# 以下函数保持原样不改，只是为了代码的完整性参考

def init_searcher(index_dir: str):
    """初始化 searcher"""
    global _searcher, _index_dir
    _index_dir = index_dir
    _searcher = LuceneSearcher(index_dir)

def get_searcher():
    """获取 searcher"""
    global _searcher
    if _searcher is None:
        raise RuntimeError("Searcher not initialized. Call init_searcher() first.")
    return _searcher


def chat_completion(client: OpenAI, messages: list[dict[str, str]], model_config: dict):
    """非流式调用"""
    completion = client.chat.completions.create(
        messages=messages,
        extra_headers={"Accept": "text/event-stream"},
        **model_config,
    )

    reasoning_content = ""
    content = ""
    try:
        reasoning_content = completion.choices[0].message.reasoning_content
    except:
        pass
    try:
        content = completion.choices[0].message.content
    except:
        pass

    return reasoning_content, content


def ask_llm(
    messages: list[dict[str, str]],
    model_config: dict,
    base_url: str,
    api_key: str,
) -> tuple[str, str]:
    """调用LLM"""
    success = False
    client = None
    for i in range(MAX_RETRIES):
        try:
            client = OpenAI(
                base_url=base_url if base_url else os.environ.get("OPENAI_BASE_URL"),
                api_key=api_key if api_key else os.environ.get("OPENAI_API_KEY"),
            )

            reasoning_content, content = chat_completion(client, messages, model_config)

            if reasoning_content or content:
                success = True
                break
            else:
                raise Exception("reasoning_content and content is empty")
        except Exception as e:
            logger.error(f"Error occurred: {model_config['model']} {e}. Retry {i+1}/{MAX_RETRIES}.")
            time.sleep(3)
        finally:
            if client:
                client.close()

    if not success:
        logger.error(f"Retry {MAX_RETRIES} but can't success!")
        reasoning_content = ""
        content = ""
    return reasoning_content, content


if __name__ == "__main__":
    import asyncio
    
    async def test_async():
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not base_url or not api_key:
            print("❌ 错误：OPENAI_BASE_URL 或 OPENAI_API_KEY 未设置")
            sys.exit(1)
        
        print("🚀 开始测试异步 ask_llm_async 函数...")
        
        test_messages = [
            {
                "role": "user",
                "content": "用中文告诉我什么是异步编程，简要回答。"
            }
        ]
        
        model_config = {
            "model": "gpt-5.1-2025-11-13-GlobalStandard",
            "stream": False,
        }
        
        print(f"📝 发送消息: {test_messages[0]['content']}")
        print(f"🤖 使用模型: {model_config['model']}")
        print("-" * 60)
        
        try:
            reasoning_content, content = await ask_llm_async(
                messages=test_messages,
                model_config=model_config,
                base_url=base_url,
                api_key=api_key,
            )
            
            print("✅ 异步调用成功！")
            print()
            
            if reasoning_content:
                print("💭 推理内容:")
                print(reasoning_content)
                print()
            
            if content:
                print("📢 模型回复:")
                print(content)
                print()
            
        except Exception as e:
            print(f"❌ 异步调用失败: {type(e).__name__}: {str(e)}")
            sys.exit(1)
        
        print("-" * 60)
        print("✨ 异步测试完成！")
    
    # 运行异步测试
    asyncio.run(test_async())

