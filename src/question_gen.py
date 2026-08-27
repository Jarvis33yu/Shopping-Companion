import os
import sys
import time
import random
import hashlib
import argparse
from typing import Any, List, Dict
from datetime import datetime, timedelta

import ujson as json
from tqdm import tqdm
from pydantic import BaseModel

from mem.retriever import ConversationDTO

import copy
from util.llm import ParallelOpenAICompletion, CompletionRequest
from prompt.question_gen import (
    single_product_question_prompt,
    add_on_deals_question_prompt,
    repeat_purchase_question_prompt,
    complement_question_prompt
)


class QuestionGen(BaseModel):
    args: Any

    single_product_preferences: List[Dict[str, Any]] = []
    add_on_deals_preferences: List[Dict[str, Any]] = []
    repeat_purchase_preferences: List[Dict[str, Any]] = []
    complement_preferences: List[Dict[str, Any]] = []
    conversations: List[ConversationDTO] = []
    conversation_preference_indices: List[Dict[str, int]] = []

    def load_preferences(self):
        start_time = time.time()

        filenames = os.listdir(self.args.preferences_dir)
        for filename in filenames:
            with open(os.path.join(self.args.preferences_dir, filename), "r") as fin:
                for line in fin:
                    preference = json.loads(line)
                    if filename == "single_product.jsonl":
                        if not preference.get("dialogue"):
                            continue
                        self.single_product_preferences.append(preference)
                    elif filename == "add_on_deals.jsonl":
                        if any(not p.get("dialogue") for p in preference["preferences"]):
                            continue
                        self.add_on_deals_preferences.append(preference)
                    elif filename == "repeat_purchase.jsonl":
                        if not preference.get("dialogue"):
                            continue
                        self.repeat_purchase_preferences.append(preference)
                    elif filename == "complement.jsonl":
                        if not preference.get("dialogue"):
                            continue
                        self.complement_preferences.append(preference)

        print(f"Loaded {len(self.single_product_preferences)} single product preferences", file=sys.stderr)
        print(f"Loaded {len(self.add_on_deals_preferences)} add on deals preferences", file=sys.stderr)
        print(f"Loaded {len(self.repeat_purchase_preferences)} repeat purchase preferences", file=sys.stderr)
        print(f"Loaded {len(self.complement_preferences)} complement preferences", file=sys.stderr)
        print(f"Total cost {time.time() - start_time: .2f} seconds.", file=sys.stderr)

    def load_longmemeval(self):
        start_time = time.time()

        with open(self.args.longmemeval_file, "r") as fin:
            jsonobj = json.load(fin)
        for item in tqdm(jsonobj, desc="Loading longmemeval: "):
            conversation = ConversationDTO(**item)
            self.conversations.append(conversation)

        print(f"Loaded {len(self.conversations)} conversations in {time.time() - start_time: .2f} seconds.", file=sys.stderr)

    def _should_repurchase(self, question_date: str, lastest_purchase_date: str, repeat_purchase_cycle: str) -> bool:

        # 解析日期
        q_date = datetime.strptime(question_date, "%Y/%m/%d (%a)")
        last_date = datetime.strptime(lastest_purchase_date, "%Y/%m/%d (%a)")
        
        # 计算天数差
        days_diff = (q_date - last_date).days
        
        # 获取周期对应的天数
        cycle_days_map = {
            "one week": 7,
            "two weeks": 14,
            "three weeks": 21,
            "one month": 30,
            "two months": 60
        }
        cycle_days = cycle_days_map.get(repeat_purchase_cycle, 30)
        
        # 判断：差值 >= cycle_days - 3
        return days_diff >= cycle_days - 3

    def _get_position(self, dates: List[datetime], date: datetime) -> int:
        if date < dates[0]:
            return 0

        if date >= dates[-1]:
            return len(dates)

        position = 0
        for i in range(1, len(dates)):
            if dates[i - 1] <= date < dates[i]:
                position = i
                break
        return position

    def _format_date(self, dt: datetime) -> str:
        return dt.strftime("%Y/%m/%d (%a)")

    def _random_datetime(self, start_dt: datetime, end_dt: datetime) -> datetime:
        if end_dt <= start_dt:
            return start_dt
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        return datetime.fromtimestamp(random.randint(start_ts, end_ts))

    def insert(self):
        # 记录每个对话已使用的偏好索引
        single_ptr = 0
        add_on_ptr = 0
        repeat_ptr = 0
        complement_ptr = 0

        for conversation in tqdm(self.conversations, desc="Inserting conversations: "):
            while True:
                single_product_preference = self.single_product_preferences[single_ptr]
                add_on_deals_preference = self.add_on_deals_preferences[add_on_ptr]
                repeat_purchase_preference = self.repeat_purchase_preferences[repeat_ptr]
                complement_preference = self.complement_preferences[complement_ptr]

                # 1. conflict check
                product_ids = [
                    single_product_preference["product_id"],
                    repeat_purchase_preference["product_id"],
                    complement_preference["product_id"],
                ]
                product_ids.extend([p["product_id"] for p in add_on_deals_preference["preferences"]])

                categories = [
                    single_product_preference["category"],
                    repeat_purchase_preference["category"],
                    complement_preference["category"],
                ]
                categories.extend([p["category"] for p in add_on_deals_preference["preferences"]])

                conflict = False
                if len(set(product_ids)) != len(product_ids):
                    conflict = True
                if len(set(categories)) != len(categories):
                    conflict = True
                
                if conflict:
                    single_ptr += 1
                    add_on_ptr += 1
                    repeat_ptr += 1
                    complement_ptr += 1
                    continue

                self.conversation_preference_indices.append({
                    "single_ptr": single_ptr,
                    "add_on_ptr": add_on_ptr,
                    "repeat_ptr": repeat_ptr,
                    "complement_ptr": complement_ptr
                })

                # reset
                sessions = []
                session_ids = []
                dates = []

                # question date
                now = datetime.now()
                question_date = now - timedelta(days=random.randint(0, 365))

                # insert repurchase
                cycle = repeat_purchase_preference["cycle"]
                cycle_days = 0
                if cycle == "one week":
                    cycle_days = 7
                elif cycle == "two weeks":
                    cycle_days = 14
                elif cycle == "three weeks":
                    cycle_days = 21
                elif cycle == "one month":
                    cycle_days = 30
                elif cycle == "two months":
                    cycle_days = 60
                else:
                    raise ValueError(f"Invalid cycle: {cycle}")

                lastest_purchase_date = question_date - timedelta(days=random.randint(cycle_days - 6, cycle_days))

                repeat_purchase_product_id = repeat_purchase_preference["product_id"]
                repeat_purchase_sessions = [repeat_purchase_preference["dialogue"][f"purchase_{i+1}"] for i in range(len(repeat_purchase_preference["dialogue"]))]
                for i, session in enumerate(repeat_purchase_sessions):
                    sessions.append(session)
                    session_ids.append(f"repeat_purchase_{repeat_purchase_product_id}_{i+1}")
                    dates.append(lastest_purchase_date - timedelta(days=(len(repeat_purchase_sessions) - i - 1) * cycle_days))

                if self.args.debug:
                    print(f"Inserting Repurchase:")
                    print(f"Question Date: {self._format_date(question_date)}")
                    print(f"Lastest Purchase Date: {self._format_date(lastest_purchase_date)}")
                    print(f"Cycle: {cycle}")
                    print(f"Session IDs: {session_ids}")
                    print(f"Dates: {[self._format_date(date) for date in dates]}")

                # insert complement
                complement_product_id = complement_preference["product_id"]
                complement_session = complement_preference["dialogue"]
                complement_date = question_date - timedelta(days=random.randint(1, 6))

                position = self._get_position(dates, complement_date)
                sessions.insert(position, complement_session)
                session_ids.insert(position, f"complement_{complement_product_id}")
                dates.insert(position, complement_date)

                if self.args.debug:
                    print(f"Inserting Complement:")
                    print(f"Position: {position}")
                    print(f"Complement Date: {self._format_date(complement_date)}")
                    print(f"Session IDs: {session_ids}")
                    print(f"Dates: {[self._format_date(date) for date in dates]}")

                # insert single product
                single_product_product_id = single_product_preference["product_id"]
                single_product_session = single_product_preference["dialogue"]
                single_product_date = self._random_datetime(now - timedelta(days=365), question_date - timedelta(days=8))

                position = self._get_position(dates, single_product_date)
                sessions.insert(position, single_product_session)
                session_ids.insert(position, f"single_product_{single_product_product_id}")
                dates.insert(position, single_product_date)

                if self.args.debug:
                    print(f"Inserting Single Product:")
                    print(f"Position: {position}")
                    print(f"Single Product Date: {self._format_date(single_product_date)}")
                    print(f"Session IDs: {session_ids}")
                    print(f"Dates: {[self._format_date(date) for date in dates]}")

                # insert add on deals
                for preference in add_on_deals_preference["preferences"]:
                    product_id = preference["product_id"]
                    session = preference["dialogue"]
                    date = self._random_datetime(now - timedelta(days=365), question_date - timedelta(days=8))

                    position = self._get_position(dates, date)
                    sessions.insert(position, session)
                    session_ids.insert(position, f"add_on_deals_{product_id}")
                    dates.insert(position, date)

                    if self.args.debug:
                            print(f"Inserting Add on Deals:")
                            print(f"Position: {position}")
                            print(f"Date: {self._format_date(date)}")
                            print(f"Session IDs: {session_ids}")
                            print(f"Dates: {[self._format_date(date) for date in dates]}")

                # insert longmemeval
                for session_id, session in zip(conversation.haystack_session_ids, conversation.haystack_sessions):
                    position = random.randint(0, len(dates))
                    if position == len(dates):
                        date = self._random_datetime(dates[position - 1], question_date)
                    elif position == 0:
                        date = self._random_datetime(now - timedelta(days=365), dates[position])
                    else:
                        date = self._random_datetime(dates[position - 1], dates[position])

                    sessions.insert(position, session)
                    dates.insert(position, date)
                    session_ids.insert(position, session_id)

                    if self.args.debug:
                            print(f"Inserting Longmemeval:")
                            print(f"Position: {position}")
                            print(f"Date: {self._format_date(date)}")
                            print(f"Session IDs: {session_ids}")
                            print(f"Dates: {[self._format_date(date) for date in dates]}")

                # set conversation attributes
                conversation.haystack_sessions = sessions
                conversation.haystack_session_ids = session_ids
                conversation.haystack_dates = [self._format_date(date) for date in dates]
                conversation.question_date = self._format_date(question_date)
                conversation.lastest_purchase_date = self._format_date(lastest_purchase_date)
                conversation.repeat_purchase_cycle = cycle
                conversation.complement_date = self._format_date(complement_date)

                single_ptr += 1
                add_on_ptr += 1
                repeat_ptr += 1
                complement_ptr += 1
                break

    def generate_question(self):
        client = ParallelOpenAICompletion(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_workers=self.args.max_workers,
            max_retries=3,
            retry_delay=1,
        )
        
        new_conversations = []
        all_llm_requests = []
        all_request_info = []
        
        # 第一遍：准备所有请求
        for i, conversation in enumerate(tqdm(self.conversations, desc="Preparing LLM requests: ")):
            indices = self.conversation_preference_indices[i]
            
            single_pref = self.single_product_preferences[indices["single_ptr"]]
            add_on_pref = self.add_on_deals_preferences[indices["add_on_ptr"]]
            repeat_pref = self.repeat_purchase_preferences[indices["repeat_ptr"]]
            complement_pref = self.complement_preferences[indices["complement_ptr"]]

            base_question_id = hashlib.md5(",".join(conversation.haystack_session_ids).encode()).hexdigest()

            # 1. single_product question
            prompt_single = single_product_question_prompt.format(product_name=single_pref["product_name"])
            if self.args.debug:
                print(f"Single product question prompt: {prompt_single}", file=sys.stderr)
            all_llm_requests.append(CompletionRequest(
                messages=[{"role": "user", "content": prompt_single}],
                model="gpt-5-2025-08-07-GlobalStandard",
                extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
            ))
            all_request_info.append({
                "conversation_idx": i,
                "question_id": f"{base_question_id}_1", 
                "type": "single_product",
                "preference": single_pref,
            })

            # 2. add_on_deals question
            add_on_product_names = [p["product_name"] for p in add_on_pref["preferences"]]
            voucher_type = add_on_pref["voucher_type"]
            v = add_on_pref['voucher']
            if voucher_type == "no voucher":
                voucher_str = "No voucher available"
            elif voucher_type == "platform":
                threshold = v["threshold"]
                discount = v["discount"]
                cap = v["cap"]
                voucher_str = f"The voucher is valid only when the total price of the products exceeds ${threshold}. 3. It provides a percentage discount of {discount} with a cap of ${cap}."
            elif voucher_type == "shop":
                threshold = v["threshold"]
                discount = v["discount"]
                cap = v["cap"]
                voucher_str = f"The voucher only applies to the products from the same shop. It is valid only when the total price of the products exceeds ${threshold}. 3. It provides a percentage discount of {discount} with a cap of ${cap}."
            else:
                raise ValueError(f"Invalid voucher type: {voucher_type}")
            budget = v['budget']

            prompt_add_on = add_on_deals_question_prompt.format(
                product_names="\n".join([f"{i+1}. {name}" for i, name in enumerate(add_on_product_names)]),
                voucher=voucher_str,
                budget=budget,
            )
            if self.args.debug:
                print(f"Add on deals question prompt: {prompt_add_on}", file=sys.stderr)
            all_llm_requests.append(CompletionRequest(
                messages=[{"role": "user", "content": prompt_add_on}],
                model="gpt-5-2025-08-07-GlobalStandard",
                extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
            ))
            all_request_info.append({
                "conversation_idx": i,
                "question_id": f"{base_question_id}_2", 
                "type": "add_on_deals",
                "preference": add_on_pref,
            })

            # 3. repeat_purchase question
            repeat_product_names = [
                single_pref["product_name"],
                repeat_pref["product_name"],
                complement_pref["product_name"],
            ]
            repeat_product_names.extend(add_on_product_names)
            random.shuffle(repeat_product_names)
            prompt_repeat = repeat_purchase_question_prompt.format(product_names="\n".join([f"{i+1}. {name}" for i, name in enumerate(repeat_product_names)]))
            if self.args.debug:
                print(f"Repeat purchase question prompt: {prompt_repeat}", file=sys.stderr)
            all_llm_requests.append(CompletionRequest(
                messages=[{"role": "user", "content": prompt_repeat}],
                model="gpt-5-2025-08-07-GlobalStandard",
                extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
            ))
            repeat_pref["should_repurchase"] = self._should_repurchase(
                conversation.question_date,
                conversation.lastest_purchase_date,
                conversation.repeat_purchase_cycle
            )

            all_request_info.append({
                "conversation_idx": i,
                "question_id": f"{base_question_id}_3", 
                "type": "repeat_purchase",
                "preference": repeat_pref,
            })

            # 4. complement question
            lastest_purchase_date = datetime.strptime(conversation.lastest_purchase_date, "%Y/%m/%d (%a)")
            question_date_dt = datetime.strptime(conversation.question_date, "%Y/%m/%d (%a)")

            # 计算天数差
            days_diff = (question_date_dt - lastest_purchase_date).days

            # 如果最后一次复购在一周内
            if days_diff <= 7:
                product_names = [complement_pref["product_name"], repeat_pref["product_name"]]
                product_ids = [complement_pref["product_id"], repeat_pref["product_id"]]
            else:
                product_names = [complement_pref["product_name"]]
                product_ids = [complement_pref["product_id"]]

            all_llm_requests.append(CompletionRequest(
                messages=[{"role": "user", "content": complement_question_prompt}],
                model="gpt-5-2025-08-07-GlobalStandard",
                extra_kwargs={"extra_headers": {"Accept": "text/event-stream"}},
            ))
            all_request_info.append({
                "conversation_idx": i,
                "question_id": f"{base_question_id}_4", 
                "type": "complement",
                "preference": {
                    "product_names": product_names,
                    "product_ids": product_ids,
                    "dialogue": complement_pref["dialogue"],
                },
            })
        
        print(f"Total LLM requests: {len(all_llm_requests)}", file=sys.stderr)
        llm_responses = client.batch_complete(all_llm_requests, verbose=True)
        
        # build new conversations
        for response_idx, response in enumerate(tqdm(llm_responses, desc="Processing LLM responses: ")):
            info = all_request_info[response_idx]
            conversation_idx = info["conversation_idx"]
            original_conversation = self.conversations[conversation_idx]
            
            new_conv = copy.deepcopy(original_conversation)
            question_type = info["type"]
            new_conv.question_id = info["question_id"]

            # 解析LLM返回的question
            question = response.content.strip().replace("```", "").replace("```json", "").replace("```plantext", "")
            if not question:
                continue
            new_conv.question = question

            # 设置answer
            new_conv.answer = info["preference"]

            # 设置question_type和answer_session_ids
            new_conv.question_type = question_type
            new_conv.answer_session_ids = [
                sid for sid in original_conversation.haystack_session_ids
                if sid.startswith(question_type)
            ]
            
            new_conversations.append(new_conv)
        
        self.conversations = new_conversations
        
        # save conversations
        with open(self.args.output_file, "w") as fout:
            for conversation in self.conversations:
                fout.write(conversation.model_dump_json() + "\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--preferences_dir", type=str, required=True)
    args.add_argument("--longmemeval_file", type=str, required=True)
    args.add_argument("--output_file", type=str, required=True)
    args.add_argument("--debug", type=str, default=False)
    args.add_argument("--max_workers", type=int, default=20)
    args = args.parse_args()

    random.seed(42)

    generator = QuestionGen(args=args)
    generator.load_preferences()
    generator.load_longmemeval()
    generator.insert()
    generator.generate_question()
