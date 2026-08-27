import random
import argparse

import ujson as json
from tqdm import tqdm

from mem.retriever import ConversationDTO


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--ref_file", type=str, required=True)
    args.add_argument("--test_size", type=int, default=100)
    args.add_argument("--valid_question_types", type=str, default="single_product,add_on_deals")
    args = args.parse_args()

    random.seed(42)

    valid_question_types = args.valid_question_types.split(",")

    convs = []
    with open(args.ref_file, "r") as fin:
        for line in tqdm(fin, desc="Splitting train and test set: "):
            item = json.loads(line.strip())
            conversation = ConversationDTO(**item)

            if conversation.question_type not in valid_question_types:
                continue

            convs.append(conversation)

    random.shuffle(convs)

    train_set = []
    test_set = []
    test_conversation_ids = set()
    for conversation in convs:
            conversation_id = conversation.question_id.rsplit("_", 1)[0]

            if len(test_conversation_ids) < args.test_size:
                test_conversation_ids.add(conversation_id)

            if conversation_id in test_conversation_ids:
                test_set.append(conversation)
            else:
                train_set.append(conversation)

    train_file = args.ref_file.replace(".jsonl", "_train.jsonl")
    with open(train_file, "w") as fout:
        train_set.sort(key=lambda x: x.question_id)
        for conv in train_set:
            fout.write(json.dumps(conv.model_dump(mode="json")) + "\n")

    test_file = args.ref_file.replace(".jsonl", "_test.jsonl")
    with open(test_file, "w") as fout:
        test_set.sort(key=lambda x: x.question_id)
        for conv in test_set:
            fout.write(json.dumps(conv.model_dump(mode="json")) + "\n")

    print(f"Split {len(train_set)} items into train set and {len(test_set)} items into test set")
