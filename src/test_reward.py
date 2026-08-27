import random
import argparse
import requests
from urllib.parse import quote_plus

import ujson as json
from tqdm import tqdm

from mem.retriever import ConversationDTO


def test_reward(args):
    references = []
    with open(args.reference_file, "r") as fin:
        for line in tqdm(fin, desc="Loading references: "):
            item = json.loads(line.strip())
            reference = ConversationDTO(**item)
            references.append(reference)

    reference = random.choice(references)

    print(f"Question ID: {reference.question_id}\nQuestion Type: {reference.question_type}\nQuestion: {reference.question}")

    # 1. Test calc_reward_mem_search
    kwargs = {
        "queries": [reference.question],
    }
    kwargs_str = quote_plus(json.dumps(kwargs))
    response = requests.get(f"{args.base_url}/?question_id={reference.question_id}&name=mem_search&kwargs={kwargs_str}")
    print(json.dumps(response.json(), indent=4))

    # 2. Test calc_reward_mem_view
    indices = []
    for answer_session_id in reference.answer_session_ids:
        session_idx = reference.haystack_session_ids.index(answer_session_id)
        index = 0
        for i, session in enumerate(reference.haystack_sessions):
            if i < session_idx:
                index += len(session)
            else:
                break
        indices.append(index)
    kwargs = {
        "indices": indices,
    }
    kwargs_str = quote_plus(json.dumps(kwargs))
    response = requests.get(f"{args.base_url}/?question_id={reference.question_id}&name=mem_view&kwargs={kwargs_str}")
    print(json.dumps(response.json(), indent=4))

    # 3. Test calc_reward_mem_summarize_by_date
    dates = []
    for answer_session_id in reference.answer_session_ids:
        ind = reference.haystack_session_ids.index(answer_session_id)
        date = reference.haystack_dates[ind]
        dates.append(date)

    kwargs = {
        "start_date": random.choice(dates),
        "offset": 7,
        "goal": reference.question,
    }
    kwargs_str = quote_plus(json.dumps(kwargs))

    response = requests.get(f"{args.base_url}/?question_id={reference.question_id}&name=mem_summarize_by_date&kwargs={kwargs_str}")
    print(json.dumps(response.json(), indent=4))

    # 4. Test calc_reward_product_search
    kwargs = {
        "query": reference.question,
    }
    kwargs_str = quote_plus(json.dumps(kwargs))
    response = requests.get(f"{args.base_url}/?question_id={reference.question_id}&name=product_search&kwargs={kwargs_str}")
    print(json.dumps(response.json(), indent=4))

    # 5. Test calc_reward_product_view
    product_ids = []
    if reference.question_type == "single_product":
        product_ids.append(reference.answer["product_id"])
    elif reference.question_type == "add_on_deals":
        product_ids.extend([p["product_id"] for p in reference.answer["preferences"]])

    kwargs = {
        "product_ids": product_ids,
    }
    kwargs_str = quote_plus(json.dumps(kwargs))
    response = requests.get(f"{args.base_url}/?question_id={reference.question_id}&name=product_view&kwargs={kwargs_str}")
    print(json.dumps(response.json(), indent=4))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--reference_file", type=str, required=True)
    args.add_argument("--base_url", type=str, default="http://0.0.0.0:5633")
    args = args.parse_args()

    test_reward(args)
