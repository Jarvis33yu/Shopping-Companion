import os
import sys
import time
import random
import argparse
from collections import defaultdict, Counter
from typing import Any, List, DefaultDict, Dict, Tuple, Set
from math import ceil, floor

import ujson as json
from tqdm import tqdm
from pydantic import BaseModel
from pyserini.search.lucene import LuceneSearcher

from prompt.dialogue_gen import (
    single_product_dialogue_prompt,
    single_product_dialogue_response_format,
    cycle_reasoning_prompt,
    repeat_purchase_dialogue_prompt,
    complement_dialogue_prompt
)
from util.misc import is_rubbish_kv
from util.llm import ParallelOpenAICompletion, CompletionRequest, parse_json_response


class DialogueGen(BaseModel):
    args: Any

    searcher: Any = None
    category2products: DefaultDict[str, List[str]] = defaultdict(list)
    shop2products: DefaultDict[str, List[str]] = defaultdict(list)
    product2category: Dict[str, str] = dict()
    used_product_ids: Set[str] = set()

    def load_searcher(self):
        start_time = time.time()
        self.searcher = LuceneSearcher(self.args.index_dir)
        print(
            f"Loaded searcher in {time.time() - start_time: .2f} seconds.",
            file=sys.stderr,
        )

    def load_products(self):
        start_time = time.time()

        category_counter = Counter()
        shop_counter = Counter()
        all_products = []
        with open(self.args.products_file, "r") as fin:
            for line in tqdm(fin, desc="Loading products: "):
                data = json.loads(line.strip())
                category = data["category"]
                shop_id = data["seller_id"]
                product_id = data["product_id"]

                splited = category.split(" - ")
                if len(splited) >= 2:
                    category = splited[1].strip()
                elif len(splited) == 1:
                    category = splited[0].strip()
                else:
                    continue

                category_counter[category] += 1
                shop_counter[shop_id] += 1
                all_products.append((category, shop_id, product_id))

        for category, shop_id, product_id in all_products:
            if category_counter[category] < self.args.min_products_per_category:
                continue

            self.category2products[category].append(product_id)
            self.shop2products[shop_id].append(product_id)
            self.product2category[product_id] = category

        for category, products in sorted(self.category2products.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"Loaded {len(products)} products for category: {category}", file=sys.stderr)

        number_of_categories = len(self.category2products)
        number_of_shops = len(self.shop2products)
        number_of_products = sum(len(products) for products in self.shop2products.values())
        print(f"Loaded {number_of_categories} categories, {number_of_shops} shops, and {number_of_products} products in {time.time() - start_time: .2f} seconds.", file=sys.stderr)

    def generate_single_product(self):
        """
        preference:
            - product_id
            - product_name
            - wanted_aspects
            - does_not_matter_aspects
            - prompt
            - dialogue
        preferences: List[preference]
        """
        # product -> aspects -> preference
        preferences = []
        pbar = tqdm(total=self.args.number_of_loops, desc="Sampling preferences for single product: ")
        while len(preferences) < self.args.number_of_loops:
            preference = self._sample_product_by_category()
            preferences.append(preference)
            pbar.update(1)

        # preference -> dialogue
        llm_requests = []
        for preference in preferences:
            product_name = preference["product_name"]
            aspects = preference["aspects"]
            prompt = single_product_dialogue_prompt.format(
                product_name=product_name,
                number_of_features=len(aspects) // 2,
                features=self._format_attributes(aspects),
            )
            preference["prompt"] = prompt
            if self.args.debug:
                print(prompt, file=sys.stderr)
            llm_requests.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-2025-08-07-GlobalStandard",
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                    response_format=single_product_dialogue_response_format,
                )
            )

        client = ParallelOpenAICompletion(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=self.args.max_workers,
            max_retries=3,
            retry_delay=1,
        )

        llm_responses = client.batch_complete(llm_requests, verbose=True)
        for preference, response in zip(preferences, llm_responses):
            if response.success:
                jsonobj = parse_json_response(response.content)
                if not jsonobj.get("wanted_features") or not jsonobj.get("does_not_matter_features"):
                    continue
                if not self._check_dialogue(jsonobj["dialogue"]):
                    continue
                preference["wanted_features"] = jsonobj["wanted_features"]
                preference["does_not_matter_features"] = jsonobj["does_not_matter_features"]
                preference["dialogue"] = jsonobj["dialogue"]
            else:
                print(response.error, file=sys.stderr)

        # save preferences
        os.makedirs(self.args.preferences_dir, exist_ok=True)
        with open(os.path.join(self.args.preferences_dir, "single_product.jsonl"), "w") as fout:
            for preference in preferences:
                if not preference.get("dialogue"):
                    continue
                fout.write(json.dumps(preference, ensure_ascii=False) + "\n")

    def generate_add_on_deals(self):
        """
        preference:
            - product_id
            - product_name
            - wanted_aspects
            - does_not_matter_aspects
            - prompt
            - dialogue
        multi_preference:
            - n
            - preferences: List[preference]
            - voucher:
                - voucher_type
                - total_price
                - threshold
                - discount
                - cap
        multi_preferences: List[multi_preference]
        """
        # product -> aspects -> preference
        multi_preferences = []
        pbar = tqdm(total=self.args.number_of_loops, desc="Sampling multi-preferences for add-on deals: ")
        while len(multi_preferences) < self.args.number_of_loops:
            # voucher_type
            voucher_type = random.choice(["platform", "shop"])

            # n
            n = random.randint(2, 3)

            # multi_preference
            multi_preference = {
                "n": n,
                "voucher_type": voucher_type,
                "preferences": [],
            }
            if voucher_type == "shop":
                multi_preference["preferences"] = self._sample_products_from_same_shop(n)
            elif voucher_type == "platform":
                for _ in range(n):
                    preference = self._sample_product_by_category()
                    multi_preference["preferences"].append(preference)
            else:
                raise ValueError(f"Invalid voucher type: {voucher_type}")

            total_price = sum(preference["price"] for preference in multi_preference["preferences"])
            if total_price < self.args.min_total_price:
                continue

            if len(multi_preference["preferences"]) < n:
                continue

            multi_preferences.append(multi_preference)
            pbar.update(1)

        # preference -> voucher(threshold, discount, cap, budget) -> dialogue
        llm_requests = []
        for multi_preference in multi_preferences:
            # voucher
            total_price = ceil(sum(preference["price"] for preference in multi_preference["preferences"]))
            threshold = random.randint(floor(total_price * 0.5), floor(total_price))
            discount = random.choice(['10%', '20%', '30%', '40%', '50%'])
            cap = random.randint(floor(threshold * 0.5), floor(threshold))
            price_after_voucher = ceil(max(total_price - cap, total_price * (1 - float(discount.strip('%')) / 100.0)))
            budget = random.randint(price_after_voucher, total_price)
            multi_preference["voucher"] = {
                "total_price": total_price,
                "threshold": threshold,
                "discount": discount,
                "cap": cap,
                "price_after_voucher": price_after_voucher,
                "budget": budget,
            }

            # dialogue generation prompt
            for preference in multi_preference["preferences"]:
                product_name = preference["product_name"]
                aspects = preference["aspects"]
                prompt = single_product_dialogue_prompt.format(
                    product_name=product_name,
                    number_of_features=len(aspects) // 2,
                    features=self._format_attributes(aspects),
                )
                preference["prompt"] = prompt
                if self.args.debug:
                    print(prompt, file=sys.stderr)
                llm_requests.append(
                    CompletionRequest(
                        messages=[{"role": "user", "content": prompt}],
                        model="gpt-5-2025-08-07-GlobalStandard",
                        extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                        response_format=single_product_dialogue_response_format,
                    )
                )

        client = ParallelOpenAICompletion(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=self.args.max_workers,
            max_retries=3,
            retry_delay=1,
        )

        llm_responses = client.batch_complete(llm_requests, verbose=True)
        i = 0
        for multi_preference in multi_preferences:
            for preference in multi_preference["preferences"]:
                response = llm_responses[i]
                i += 1
                if response.success:
                    jsonobj = parse_json_response(response.content)
                    if not jsonobj.get("wanted_features") or not jsonobj.get("does_not_matter_features"):
                        continue
                    if not self._check_dialogue(jsonobj["dialogue"]):
                        continue
                    preference["wanted_features"] = jsonobj["wanted_features"]
                    preference["does_not_matter_features"] = jsonobj["does_not_matter_features"]
                    preference["dialogue"] = jsonobj["dialogue"]
                else:
                    print(response.error, file=sys.stderr)

        # save multi-preferences
        os.makedirs(self.args.preferences_dir, exist_ok=True)
        with open(os.path.join(self.args.preferences_dir, "add_on_deals.jsonl"), "w") as fout:
            for multi_preference in multi_preferences:
                if not any(p.get("dialogue") for p in multi_preference["preferences"]):
                    continue
                fout.write(json.dumps(multi_preference, ensure_ascii=False) + "\n")

    def generate_repeat_purchase(self):
        preferences = []

        fast_moving_categories = [
            "Personal Care",
            "Skin Care",
            "Snacks & Confectionery",
            "Drinks",
            "Makeup",
            "Food Supplement",
            "Pet Food",
            "Fragrances",
            "Cleaning Agents",
            "Paper Products",
            "Laundry Supplies",
            "Diapering & Potty",
            "Milk Formula & Baby Food",
            "Dairy Chilled & Eggs",
            "Fruit & Vegetables",
            "Meat & Seafood",
            "Frozen",
            "Bakery & Breakfast",
            "Alcoholic Beverages",
            "Paper & Tissue",
        ]
 
        pbar = tqdm(total=self.args.number_of_loops, desc="Sampling repeat purchase preferences: ")
        while len(preferences) < self.args.number_of_loops:
            # ============ 第1步：随机采样1件商品 ============
            category = random.choice(fast_moving_categories)
            if len(self.category2products[category]) == 0:
                continue
            product_id = random.choice(self.category2products[category])

            if product_id in self.used_product_ids:
                continue

            doc = self.searcher.doc(product_id)
            if not doc:
                continue

            self.used_product_ids.add(product_id)

            product = json.loads(doc.raw())["product"]
            product_name = product["product_name"]
            price = product["price"]

            # ============ 第2步：建立偏好对象 ============
            preference = {
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "price": price,
            }
            preferences.append(preference)
            pbar.update(1)

        pbar.close()

        # ============ 第3步：批量LLM调用 - 推理复购周期 ============
        llm_requests_cycle = []
        for preference in preferences:
            product_name = preference["product_name"]
            category = preference["category"]
            price = preference["price"]

            prompt = cycle_reasoning_prompt.format(product_name=product_name, category=category, price=price)
            if self.args.debug:
                print(prompt, file=sys.stderr)

            llm_requests_cycle.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-2025-08-07-GlobalStandard",
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                )
            )

        client = ParallelOpenAICompletion(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=self.args.max_workers,
            max_retries=3,
            retry_delay=1,
        )

        cycle_responses = client.batch_complete(llm_requests_cycle, verbose=True)

        valid_preferences = []
        for preference, response in zip(preferences, cycle_responses):
            if response.success:
                jsonobj = parse_json_response(response.content)

                if not jsonobj:
                    continue

                cycle = jsonobj.get("cycle").strip()
                cycle_think = jsonobj.get("think")

                if cycle not in {"one week", "two weeks", "three weeks", "one month", "two months"}:
                    continue

                preference["cycle"] = cycle
                preference["cycle_think"] = cycle_think
                valid_preferences.append(preference)
            else:
                print(response.error, file=sys.stderr)

        preferences = valid_preferences

        # ============ 第4步：批量LLM调用 - 生成对话 ============
        llm_requests_dialogue = []
        for preference in preferences:
            product_name = preference["product_name"]
            category = preference["category"]
            price = preference["price"]
            cycle = preference["cycle"]

            prompt = repeat_purchase_dialogue_prompt.format(product_name=product_name, category=category, price=price, cycle=cycle)

            llm_requests_dialogue.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-2025-08-07-GlobalStandard",
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                )
            )

        dialogue_responses = client.batch_complete(llm_requests_dialogue, verbose=True)

        valid_preferences = []
        for preference, response in zip(preferences, dialogue_responses):
            if response.success:
                dialogue = parse_json_response(response.content)

                is_success = True
                for k, v in dialogue.items():
                    if not k.startswith("purchase_") or len(v) == 0:
                        is_success = False
                        break
                    if not self._check_dialogue(v):
                        is_success = False
                        break
                if not is_success:
                    continue

                preference["dialogue"] = dialogue
                valid_preferences.append(preference)
            else:
                print(response.error, file=sys.stderr)

        preferences = valid_preferences

        # ============ 保存结果 ============
        with open(os.path.join(self.args.preferences_dir, "repeat_purchase.jsonl"), "w") as fout:
            for preference in preferences:
                fout.write(json.dumps(preference, ensure_ascii=False) + "\n")
 
        print(
            f"Successfully generated {len(preferences)} repeat purchase preferences.",
            file=sys.stderr,
        )

    def generate_complement(self):
        preferences = []
 
        pbar = tqdm(total=self.args.number_of_loops, desc="Sampling complement purchase preferences: ")
        while len(preferences) < self.args.number_of_loops:
            # ============ 第1步：随机采样1件商品 ============
            preference = self._sample_product_by_category()

            # ============ 第2步：商品本身即为用户偏好 ============
            preferences.append(preference)
            pbar.update(1)

        pbar.close()

        # ============ 第3步：生成对话 ============
        llm_requests = []
        for preference in preferences:
            product_name = preference["product_name"]
            category = preference["category"]
            price = preference["price"]

            prompt = complement_dialogue_prompt.format(product_name=product_name, category=category, price=price)
            if self.args.debug:
                print(prompt, file=sys.stderr)

            llm_requests.append(
                CompletionRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-5-2025-08-07-GlobalStandard",
                    extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
                )
            )

        # 批量调用LLM生成对话
        client = ParallelOpenAICompletion(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=self.args.max_workers,
            max_retries=3,
            retry_delay=1,
        )

        llm_responses = client.batch_complete(llm_requests, verbose=True)

        # 映射响应到偏好对象
        valid_preferences = []
        for preference, response in zip(preferences, llm_responses):
            if response.success:
                dialogue = parse_json_response(response.content)

                if not self._check_dialogue(dialogue):
                    continue

                preference["dialogue"] = dialogue
                valid_preferences.append(preference)
            else:
                print(response.error, file=sys.stderr)

        preferences = valid_preferences

        # ============ 保存结果 ============
        with open(os.path.join(self.args.preferences_dir, "complement.jsonl"), "w") as fout:
            for preference in preferences:
                fout.write(json.dumps(preference, ensure_ascii=False) + "\n")

        print(
            f"Successfully generated {len(preferences)} complement purchase preferences.",
            file=sys.stderr,
        )

    def _confuse_price(self, price: float) -> Tuple[str, str]:
        confusions = ["less than", "greater than", "between"]
        confusion = random.choice(confusions)
        if confusion == "less than":
            lower_bound = 0
            upper_bound = int(price + random.randint(1, ceil(price)))
            text = f"less than ${upper_bound}"
        elif confusion == "greater than":
            lower_bound = int(price - random.randint(1, ceil(price)))
            upper_bound = None
            text = f"greater than ${lower_bound}"
        else:
            lower_bound = int(price - random.randint(1, ceil(price)))
            upper_bound = int(price + random.randint(1, ceil(price)))
            text = f"between ${lower_bound} and ${upper_bound}"

        if lower_bound < 0 or price <= lower_bound:
            return None
        if upper_bound is not None and price >= upper_bound:
            return None

        return "price", text

    def _format_attributes(self, attributes: List[Tuple[str, str]]) -> str:
        if not attributes:
            return ""
        results = []
        for attribute in attributes:
            if len(attribute) != 2:
                continue
            k = attribute[0]
            v = attribute[1]
            results.append(f"{k}: {v}")
        return "\n".join(results)

    def _sample_product_by_category(self) -> Dict[str, Any]:
        while True:
            category = random.choice(list(self.category2products.keys()))
            if len(self.category2products[category]) == 0:
                continue
            product_id = random.choice(self.category2products[category])

            if product_id in self.used_product_ids:
                continue

            doc = self.searcher.doc(product_id)
            if not doc:
                continue

            self.used_product_ids.add(product_id)

            product = json.loads(doc.raw())["product"]
            product_name = product["product_name"]
            price = product["price"]
            attributes = product.get("attributes")
            options = product.get("options")

            aspects = set()
            if attributes:
                for k, vs in attributes.items():
                    for v in vs:
                        if is_rubbish_kv(k, v):
                            continue
                        aspects.add((k, v))
            if options:
                option = random.choice(options)
                for k, vs in option.items():
                    for v in vs:
                        if is_rubbish_kv(k, v):
                            continue
                        aspects.add((k, v))

            if not 3 <= len(aspects) <= 8:
                continue

            aspects = list(aspects)
            random.shuffle(aspects)

            preference = {
                "product_id": product_id,
                "product_name": product_name,
                "price": price,
                "category": category,
                "aspects": aspects,
            }

            return preference

    def _sample_products_from_same_shop(self, n: int) -> List[Dict[str, Any]]:
        while True:
            shop_id = random.choice(list(self.shop2products.keys()))
            product_ids = self.shop2products[shop_id]
            if len(product_ids) < n:
                continue
            c2p = defaultdict(list)
            for product_id in product_ids:
                category = self.product2category[product_id]
                c2p[category].append(product_id)
            if len(c2p) < n:
                continue

            categories = random.sample(list(c2p.keys()), n)
            selected_product_ids = []
            for category in categories:
                selected_product_ids.append(random.choice(c2p[category]))

            if any(product_id in self.used_product_ids for product_id in selected_product_ids):
                continue

            if any(not self.searcher.doc(product_id) for product_id in selected_product_ids):
                continue

            products = [
                json.loads(self.searcher.doc(product_id).raw())["product"]
                for product_id in selected_product_ids
            ]

            if len(products) < n:
                continue

            self.used_product_ids.update(selected_product_ids)

            multi_preference = []
            for product in products:
                product_id = product["product_id"]
                product_name = product["product_name"]
                price = product["price"]
                shop_id = product["seller_id"]
                attributes = product.get("attributes")
                options = product.get("options")
                category = self.product2category[product_id]

                aspects = set()
                if attributes:
                    for k, vs in attributes.items():
                        for v in vs:
                            if is_rubbish_kv(k, v):
                                continue
                            aspects.add((k, v))
                if options:
                    option = random.choice(options)
                    for k, vs in option.items():
                        for v in vs:
                            if is_rubbish_kv(k, v):
                                continue
                            aspects.add((k, v))

                if not 3 <= len(aspects) <= 8:
                    continue

                aspects = list(aspects)
                random.shuffle(aspects)

                preference = {
                    "product_id": product_id,
                    "product_name": product_name,
                    "price": price,
                    "shop_id": shop_id,
                    "category": category,
                    "aspects": aspects,
                }
                multi_preference.append(preference)

            if len(multi_preference) < n:
                continue

            return multi_preference

    def _check_dialogue(self, dialogue: List[Dict[str, Any]]) -> bool:
        if len(dialogue) == 0:
            return False

        for turn in dialogue:
            if not turn.get("role") or not turn.get("content"):
                return False
            if turn["role"] not in {"user", "assistant"}:
                return False
            if len(turn["content"]) == 0:
                return False
        return True


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--index_dir", type=str, required=True)
    args.add_argument("--products_file", type=str, required=True)
    args.add_argument("--preferences_dir", type=str, required=True)
    args.add_argument("--number_of_loops", type=int, default=1000)
    args.add_argument("--max_workers", type=int, default=20)
    args.add_argument("--debug", type=bool, default=False)
    args.add_argument("--min_products_per_category", type=int, default=1000)
    args.add_argument("--min_total_price", type=int, default=10)
    args = args.parse_args()

    random.seed(42)

    generator = DialogueGen(args=args)
    generator.load_searcher()
    generator.load_products()
    generator.generate_single_product()
    generator.generate_add_on_deals()
    generator.generate_repeat_purchase()
    generator.generate_complement()
