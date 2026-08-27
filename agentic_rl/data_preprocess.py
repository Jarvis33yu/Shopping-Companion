import argparse
import json
import os
import hashlib
from typing import List, Dict, Any
import random
import pandas as pd


two_stage_1_prompt = """# Instructions

## Task
Given a product search query, retrieve the user's relevant memories (dialogue history) and identify their purchase preferences from them.

You can complete the task by:
- Retrieve the most relevant memories (dialogue turns) using the "mem_search" tool.
- View the whole dialogue session using the "mem_view" tool.
- Summarize dialogue sessions by date range using the "mem_summarize_by_date" tool.

## Rules
In each turn you can either:
- Think and make one or more tool calls.
- Provide your final answer and terminate the conversation.
You cannot do both at the same time.

You MUST think step by step and make multi-turn tool calls before providing your final answer.

# Output Format

## For thinking and making tool calls
Format the output as follows:
- Reasoning process MUST be within <think></think> tags.
- Tool calls MUST be within <tool_call></tool_call> tags. Each line MUST be a valid JSON object with "name" and "arguments" fields.
- Strictly follow the template below (DO NOT FORGET THE </tool_call> TAG):
```plaintext
<think>...your reasoning process...

</think>
<tool_call>
{"name": "mem_search", "arguments": {"queries": ["...", "..."]}}
...

</tool_call>
```

## For providing your final answer
Write concise content within <answer></answer> tags as follows:
- List the relevant parts of memories and identified preferences in human-readable format.
- Ask if the above information is sufficient and accurate.
- Strictly follow the template below:
```plaintext
<answer>...your concise content...

</answer>
```
"""


two_stage_2_prompt = """# Instructions

## Task
Given the product search query and the user's purchase preferences, find products or product bundles that exactly match them.

You can complete the task by:
- Use the "product_search" tool to search for products. Do not recommend any products from your own knowledge base.
- Use the "product_view" tool to view the attributes and options of the products, and than check if they match the user's preferences.
- Obtain up-to-date or domain-specific knowledge from the Internet using the "web_search" tool, and then visit and summarize the webpages using the "web_visit" tool.

## Rules
In each turn you can either:
- Think and make one or more tool calls.
- Provide your final answer and terminate the conversation.
You cannot do both at the same time.

You MUST think step by step and make multi-turn tool calls before providing your final answer.

# Output Format

## For thinking and making tool calls
Format the output as follows:
- Reasoning process MUST be within <think></think> tags.
- Tool calls MUST be within <tool_call></tool_call> tags. Each line MUST be a valid JSON object with "name" and "arguments" fields.
- Strictly follow the template below (DO NOT FORGET THE </tool_call> TAG):
```plaintext
<think>...your reasoning process...

</think>
<tool_call>
{"name": "product_search", "arguments": {"query": "...", "shop_id": "...", "price": "..."}}
...

</tool_call>
```

## For providing your final answer
Write a expert-level report within <answer></answer> tags as follows:
- Describe how the products or product bundles you found align with user's preferences.
- Enclose the best-matching recommendation in the **special format**: @REC::product_id@ for a single product, or @REC::product_id1,product_id2,...@ for a product bundle.
- The product_id in the **special format** MUST come from the "product_search" tool response.
- May discuss multiple products or bundles, but only use the **special format** once.
- Is formatted with clear, well-structured Markdown.
- Strictly follow the template below:
```plaintext
<answer>...your expert-level report...

</answer>
```
"""

def compute_conversation_id(haystack_session_ids: List[str]) -> str:
    """Compute conversation_id as MD5 hash of sorted session IDs joined by comma."""
    return hashlib.md5(",".join(haystack_session_ids).encode()).hexdigest()

def extract_preferences_string(example: Dict[str, Any]) -> str:

    answer = example.get("answer", {})
    question_type = example.get("question_type", "")
    
    if question_type == "single_product":
        wanted_features = answer.get("wanted_features", [])
        does_not_matter_features = answer.get("does_not_matter_features", [])
        return _format_single_preference(wanted_features, does_not_matter_features)
    
    elif question_type == "add_on_deals":
        preferences_list = answer.get("preferences", [])
        preference_strings = []
        
        for idx, pref in enumerate(preferences_list, 1):
            wanted_features = pref.get("wanted_features", [])
            does_not_matter_features = pref.get("does_not_matter_features", [])
            pref_str = _format_single_preference(wanted_features, does_not_matter_features)
            preference_strings.append(f"Product {idx}: {pref_str}")
        
        return "; ".join(preference_strings)
    
    return ""


def _format_single_preference(wanted_features: list, 
                              does_not_matter_features: list) -> str:
    """
    Format a single product's preference information
    """
    parts = []
    
    # Add wanted features
    if wanted_features:
        features_str = ", ".join(wanted_features)
        parts.append(f"Wanted features: {features_str}")
    
    # Add does_not_matter features
    if does_not_matter_features:
        features_str = ", ".join(does_not_matter_features)
        parts.append(f"Indifferent features: {features_str}")
    
    return "; ".join(parts) if parts else "No specific preferences"


def build_example(example: Dict[str, Any], idx: int, split: str, prompt_version: int) -> Dict[str, Any]:
    """Build a single example with specified prompt version."""
    question = example["question"]
    question_date = example["question_date"]
    question = f"Current Date: {question_date}\n{question}"
    
    answer = example["answer"]
    
    question_id = example.get("question_id", "")
    question_type = example.get("question_type", "")
    haystack_session_ids = example.get("haystack_session_ids", [])
    
    if not isinstance(haystack_session_ids, list):
        haystack_session_ids = [haystack_session_ids] if haystack_session_ids else []
    
    conversation_id = compute_conversation_id(haystack_session_ids) if haystack_session_ids else ""


    # oracle
    oracle = []
    answer_session_ids = example.get("answer_session_ids", [])
    haystack_sessions = example.get("haystack_sessions", [])
    haystack_dates = example.get("haystack_dates", [])
    for answer_session_id in answer_session_ids:
        for session_id, session, date in zip(
            haystack_session_ids,
            haystack_sessions,
            haystack_dates,
        ):
            if session_id == answer_session_id:
                session_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in session])
                oracle.append(f"[Date: {date}]\n{session_str}")
    oracle_str = "The most relevant user dialogue memories are as follows:\n\n" + "\n\n".join(oracle)


    # Select prompt based on version
    if prompt_version == 1:
        system_prompt = two_stage_1_prompt  
        data = {
            "data_source": "shopping_companion",
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "ability": "stage_1",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(answer, ensure_ascii=False) if not isinstance(answer, str) else answer
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "ability": "stage_1",
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "conversation_id": conversation_id,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "mem_search": {
                        "create_kwargs": {
                            "conversation_id": conversation_id,
                            "question_id": question_id
                            },
                    },
                    "mem_view": {
                        "create_kwargs": {
                            "conversation_id": conversation_id,
                            "question_id": question_id
                            },
                    },
                    "mem_summarize_by_date": {
                        "create_kwargs": {
                            "conversation_id": conversation_id,
                            "question_id": question_id
                            },
                    },
                    "product_search": {
                        "create_kwargs": {"question_id": question_id}
                    },
                    "product_view": {
                        "create_kwargs": {"question_id": question_id}
                    },
                }
            },
        }
    else:  # prompt_version == 2
        system_prompt = two_stage_2_prompt
        preference = extract_preferences_string(example)
        data = {
            "data_source": "shopping_companion",
            "agent_name": "tool_agent",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "user",
                    "content": oracle_str,
                },
                {
                    "role": "user",
                    "content": "I will not provide you with any more information. Please complete the task.",
                },
            ],
            "ability": "stage_2",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(answer, ensure_ascii=False) if not isinstance(answer, str) else answer
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "ability": "stage_2",
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "conversation_id": conversation_id,
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "mem_search": {
                        "create_kwargs": {
                            "conversation_id": conversation_id,
                            "question_id": question_id
                            },
                    },
                    "mem_view": {
                        "create_kwargs": {
                            "conversation_id": conversation_id,
                            "question_id": question_id
                            },
                    },
                    "mem_summarize_by_date": {
                        "create_kwargs": {
                            "conversation_id": conversation_id,
                            "question_id": question_id
                            },
                    },
                    "product_search": {
                        "create_kwargs": {"question_id": question_id}
                    },
                    "product_view": {
                        "create_kwargs": {"question_id": question_id}
                    },
                }
            },
        }
    return data


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_dataset(examples: List[Dict[str, Any]], output_dir: str, split_name: str):
    """Save as JSON and Parquet"""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{split_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {split_name} JSON: {json_path}")

   
    try:
        df = pd.DataFrame(examples)
        parquet_path = os.path.join(output_dir, f"{split_name}.parquet")
        df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"Saved {split_name} Parquet: {parquet_path}")
    except Exception as e:
        print(f"Error saving Parquet: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Preprocess custom memory QA dataset (JSONL format) with dual prompts.")
    parser.add_argument("--input_jsonl", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output_dir", default="./preprocessed_data", help="Output directory for processed data.")
    parser.add_argument("--split_name", required=True, help="Split name (e.g., 'train', 'test').")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible shuffle.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy results to.")

    args = parser.parse_args()

    # Load raw data from JSONL
    raw_data = load_jsonl(args.input_jsonl)
    print(f"Loaded {len(raw_data)} examples from {args.input_jsonl}")

    # Process examples with both prompt versions
    print("\nProcessing examples with dual prompts...")
    all_examples = []
    for idx, ex in enumerate(raw_data):
        # Create example with prompt version 1
        example_v1 = build_example(ex, idx, args.split_name, prompt_version=1)
        all_examples.append(example_v1)
        
        # Create example with prompt version 2
        example_v2 = build_example(ex, idx, args.split_name, prompt_version=2)
        all_examples.append(example_v2)
    
    print(f"Processed {len(raw_data)} examples into {len(all_examples)} examples (dual prompts)")

    # Shuffle all_examples
    random.seed(args.random_seed)
    random.shuffle(all_examples)
    print(f"Shuffled {len(all_examples)} examples with random seed: {args.random_seed}")


    # Save locally
    print("\nSaving dataset...")
    save_dataset(all_examples, args.output_dir, args.split_name)

    # Optional: upload to HDFS
    if args.hdfs_dir:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            print(f"\nUploading to HDFS: {args.hdfs_dir}")
            makedirs(args.hdfs_dir)
            copy(src=args.output_dir, dst=args.hdfs_dir)
            print(f"Copied results to HDFS: {args.hdfs_dir}")
        except ImportError:
            print("Warning: HDFS utilities not available. Skipping HDFS upload.")

    print("\nPreprocessing completed!")


if __name__ == "__main__":
    main()
