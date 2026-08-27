import sys

import ujson as json
import tiktoken
import numpy as np

rollout_file = sys.argv[1]

enc = tiktoken.get_encoding("cl100k_base")

prompt_tokens_list = []
completion_tokens_list = []

with open(rollout_file, "r") as fin:
    for line in fin:
        item = json.loads(line.strip())
        if not item:
            continue
        
        messages = item[-1]["prompt"]
        completion = item[-1]["reward_model"]["ground_truth"]
        prompt_tokens = 0
        for message in messages:
            prompt_tokens += len(enc.encode(message["content"]))
        completion_tokens = len(enc.encode(completion))
        
        prompt_tokens_list.append(prompt_tokens)
        completion_tokens_list.append(completion_tokens)
        
# 统计信息
if prompt_tokens_list and completion_tokens_list:
    print("\n" + "=" * 80)
    print("统计信息:")
    print("=" * 80)
    
    print("\nPrompt Tokens 统计:")
    print(f"  样本数量: {len(prompt_tokens_list)}")
    print(f"  平均值: {np.mean(prompt_tokens_list):.2f} ({np.mean(prompt_tokens_list) / 1024:.3f}k)")
    print(f"  最小值: {np.min(prompt_tokens_list)} ({np.min(prompt_tokens_list) / 1024:.3f}k)")
    print(f"  最大值: {np.max(prompt_tokens_list)} ({np.max(prompt_tokens_list) / 1024:.3f}k)")
    print(f"  中位数 (50%): {np.percentile(prompt_tokens_list, 50):.2f} ({np.percentile(prompt_tokens_list, 50) / 1024:.3f}k)")
    print(f"  25% 分位数: {np.percentile(prompt_tokens_list, 25):.2f} ({np.percentile(prompt_tokens_list, 25) / 1024:.3f}k)")
    print(f"  75% 分位数: {np.percentile(prompt_tokens_list, 75):.2f} ({np.percentile(prompt_tokens_list, 75) / 1024:.3f}k)")
    print(f"  90% 分位数: {np.percentile(prompt_tokens_list, 90):.2f} ({np.percentile(prompt_tokens_list, 90) / 1024:.3f}k)")
    print(f"  95% 分位数: {np.percentile(prompt_tokens_list, 95):.2f} ({np.percentile(prompt_tokens_list, 95) / 1024:.3f}k)")
    print(f"  99% 分位数: {np.percentile(prompt_tokens_list, 99):.2f} ({np.percentile(prompt_tokens_list, 99) / 1024:.3f}k)")
    
    print("\nCompletion Tokens 统计:")
    print(f"  样本数量: {len(completion_tokens_list)}")
    print(f"  平均值: {np.mean(completion_tokens_list):.2f} ({np.mean(completion_tokens_list) / 1024:.3f}k)")
    print(f"  最小值: {np.min(completion_tokens_list)} ({np.min(completion_tokens_list) / 1024:.3f}k)")
    print(f"  最大值: {np.max(completion_tokens_list)} ({np.max(completion_tokens_list) / 1024:.3f}k)")
    print(f"  中位数 (50%): {np.percentile(completion_tokens_list, 50):.2f} ({np.percentile(completion_tokens_list, 50) / 1024:.3f}k)")
    print(f"  25% 分位数: {np.percentile(completion_tokens_list, 25):.2f} ({np.percentile(completion_tokens_list, 25) / 1024:.3f}k)")
    print(f"  75% 分位数: {np.percentile(completion_tokens_list, 75):.2f} ({np.percentile(completion_tokens_list, 75) / 1024:.3f}k)")
    print(f"  90% 分位数: {np.percentile(completion_tokens_list, 90):.2f} ({np.percentile(completion_tokens_list, 90) / 1024:.3f}k)")
    print(f"  95% 分位数: {np.percentile(completion_tokens_list, 95):.2f} ({np.percentile(completion_tokens_list, 95) / 1024:.3f}k)")
    print(f"  99% 分位数: {np.percentile(completion_tokens_list, 99):.2f} ({np.percentile(completion_tokens_list, 99) / 1024:.3f}k)")
    
    print("\n总 Tokens 统计:")
    total_tokens_list = [p + c for p, c in zip(prompt_tokens_list, completion_tokens_list)]
    print(f"  样本数量: {len(total_tokens_list)}")
    print(f"  平均值: {np.mean(total_tokens_list):.2f} ({np.mean(total_tokens_list) / 1024:.3f}k)")
    print(f"  最小值: {np.min(total_tokens_list)} ({np.min(total_tokens_list) / 1024:.3f}k)")
    print(f"  最大值: {np.max(total_tokens_list)} ({np.max(total_tokens_list) / 1024:.3f}k)")
    print(f"  中位数 (50%): {np.percentile(total_tokens_list, 50):.2f} ({np.percentile(total_tokens_list, 50) / 1024:.3f}k)")
    print(f"  25% 分位数: {np.percentile(total_tokens_list, 25):.2f} ({np.percentile(total_tokens_list, 25) / 1024:.3f}k)")
    print(f"  75% 分位数: {np.percentile(total_tokens_list, 75):.2f} ({np.percentile(total_tokens_list, 75) / 1024:.3f}k)")
    print(f"  90% 分位数: {np.percentile(total_tokens_list, 90):.2f} ({np.percentile(total_tokens_list, 90) / 1024:.3f}k)")
    print(f"  95% 分位数: {np.percentile(total_tokens_list, 95):.2f} ({np.percentile(total_tokens_list, 95) / 1024:.3f}k)")
    print(f"  99% 分位数: {np.percentile(total_tokens_list, 99):.2f} ({np.percentile(total_tokens_list, 99) / 1024:.3f}k)")
    print("=" * 80)
