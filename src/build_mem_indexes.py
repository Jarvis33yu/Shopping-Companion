import hashlib
import argparse

import ujson as json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from mem.retriever import ConversationDTO, Retriever


class BuildMemIndexes:
    def __init__(self, args):
        self.args = args

    def build_indexes(self):
        if self.args.conversation_file.endswith(".json"):
            with open(self.args.conversation_file, "r") as fin:
                conversations = json.load(fin)
        elif self.args.conversation_file.endswith(".jsonl"):
            with open(self.args.conversation_file, "r") as fin:
                conversations = [json.loads(line) for line in fin]
        else:
            raise ValueError(f"Unsupported file extension: {self.args.conversation_file}")

        # conversations
        conversations = [ConversationDTO(**conversation) for conversation in conversations]

        # retriever
        sentence_model = SentenceTransformer(self.args.sentence_model_name)
        conversation_ids = set()
        for conversation in tqdm(conversations, desc="Building indexes: "):
            conversation_id = hashlib.md5(",".join(conversation.haystack_session_ids).encode()).hexdigest()
            if conversation_id in conversation_ids:
                continue
            conversation_ids.add(conversation_id)
            retriever = Retriever(sentence_model=sentence_model)
            retriever.build_index(conversation_id=conversation_id, conversation=conversation)
            retriever.save_index(index_dir=self.args.index_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--conversation_file", type=str, required=True)
    args.add_argument("--index_dir", type=str, required=True)
    args.add_argument("--sentence_model_name", type=str, default="all-MiniLM-L6-v2")
    args = args.parse_args()

    builder = BuildMemIndexes(args)
    builder.build_indexes()
