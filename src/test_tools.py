import os
import sys
import random

import ujson as json
from tqdm import tqdm

from tools import all_tools


def test_tools(tool_type):
    if tool_type == "mem":
        ref_file = "data/shopping_companion_s_cleaned_test.jsonl"
        items = []
        if os.path.exists(ref_file):
            with open(ref_file, "r") as fin:
                for line in tqdm(fin, desc="Loading memory file: "):
                    items.append(json.loads(line.strip()))

        item = random.choice(items)

        conversation_id = item["question_id"].rsplit("_", 1)[0]
        queries = [item["question"]]
        goal = item["question"]
        memory_indices = [0, 100, 1000, 10000]
        start_date = random.choice(item["haystack_dates"])
        offset = 7

    for toolclass in all_tools:
        tool = toolclass()

        if tool_type == "web" and tool.name == "web_search":
            print(tool.to_string())
            print("\n")
            tool_response = tool.execute(queries=["How to learn Python", "What is policy gradient"])
            print(tool_response)
            print("\n")
        elif tool_type == "web" and tool.name == "web_visit":
            print(tool.to_string())
            print("\n")
            tool_response = tool.execute(
                urls=[
                    "https://www.runoob.com/python/python-tutorial.html",
                    "https://www.baidu.com/",
                    "https://shopping.companion.agent",
                ],
                goal="Search for Python tutorials",
            )
            print(tool_response)
            print("\n")
        elif tool_type == "product" and tool.name == "product_search":
            print(tool.to_string())
            print("\n")
            tool_response = tool.execute(query="iphone 17", price="500-")
            print(tool_response)
            print("\n")
        elif tool_type == "product" and tool.name == "product_view":
            print(tool.to_string())
            print("\n")
            tool_response = tool.execute(product_ids=["5250901107", "5250923607", "5251048584"])
            print(tool_response)
            print("\n")
        elif tool_type == "mem" and tool.name == "mem_search":
            print(tool.to_string())
            print("\n")
            tool_response = tool.execute(queries=queries, conversation_id=conversation_id)
            print(tool_response)
            print("\n")
        elif tool_type == "mem" and tool.name == "mem_view":
            print(tool.to_string())
            print("\n")
            tool_response = tool.execute(indices=memory_indices, conversation_id=conversation_id)
            print(tool_response)
            print("\n")
        elif tool_type == "mem" and tool.name == "mem_summarize_by_date":
            print(tool.to_string())
            print("\n")
            tool_response = tool.execute(start_date=start_date, offset=offset, goal=goal, conversation_id=conversation_id)
            print(tool_response)
            print("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_tools.py <tool_type>")
        sys.exit(1)
    tool_type = sys.argv[1]

    test_tools(tool_type)
