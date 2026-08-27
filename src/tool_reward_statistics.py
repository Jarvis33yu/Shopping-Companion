#!/usr/bin/env python3
"""
统计不同工具调用的平均 reward 分数

从 data/* 中的 rollout 文件读取轨迹数据，解析每个 step 中的 tool_calls，
调用 serve_reward 服务获取 reward 分数，最后统计各个工具的平均 reward。
"""

import os
import re
import sys
import argparse
from typing import Dict, List, Any, Optional
from collections import defaultdict

# 尝试使用 ujson，如果不可用则使用标准 json
try:
    import ujson as json
except ImportError:
    import json

try:
    from tqdm import tqdm
except ImportError:
    # 如果 tqdm 不可用，提供一个简单的替代
    def tqdm(iterable, **kwargs):
        return iterable


def extract_tool_calls(ground_truth: str) -> List[Dict[str, Any]]:
    """
    从 ground_truth 中提取 tool_calls
    逻辑参考 agent_loop.py 的 act 方法
    """
    tool_calls = []
    
    # 使用正则提取所有 <tool_call>...</tool_call> 块
    tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', ground_truth, re.DOTALL)
    
    for tool_call_block in tool_call_blocks:
        tool_call_block = tool_call_block.strip()
        for tool_call in tool_call_block.split("\n"):
            tool_call = tool_call.strip()
            if not tool_call:
                continue
            
            try:
                jsonobj = json.loads(tool_call)
                name = jsonobj.get("name")
                arguments = jsonobj.get("arguments")
                
                if name and arguments and isinstance(arguments, dict):
                    tool_calls.append({
                        "name": name,
                        "arguments": arguments
                    })
            except Exception as e:
                # 忽略解析失败的 tool_call
                continue
    
    return tool_calls


def call_reward_service(
    reward_url: str,
    question_id: str,
    tool_name: str,
    tool_arguments: Dict[str, Any]
) -> Optional[float]:
    """
    调用 serve_reward 服务获取 reward 分数
    """
    try:
        import requests  # 延迟导入，仅在需要时导入
        
        params = {
            "question_id": question_id,
            "name": tool_name,
            "kwargs": json.dumps(tool_arguments)
        }
        
        response = requests.get(reward_url, params=params, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("reward", 0.0)
        else:
            print(f"Warning: Failed to get reward for {tool_name}: HTTP {response.status_code}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Warning: Error calling reward service for {tool_name}: {e}", file=sys.stderr)
        return None


def process_rollout_file(
    filepath: str,
    reward_url: str,
    verbose: bool = False,
    dry_run: bool = False
) -> Dict[str, List[float]]:
    """
    处理单个 rollout 文件，返回各个工具的 reward 列表
    
    Args:
        filepath: rollout 文件路径
        reward_url: reward 服务 URL
        verbose: 是否显示进度条
        dry_run: 是否为干运行模式（不调用 reward 服务）
    """
    tool_rewards = defaultdict(list)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    iterator = tqdm(lines, desc=f"Processing {os.path.basename(filepath)}") if verbose else lines
    
    for line in iterator:
        try:
            trajectory = json.loads(line.strip())
            if not trajectory:
                continue
            
            # 遍历轨迹中的每个 step
            for step in trajectory:
                ground_truth = step.get("reward_model", {}).get("ground_truth", "")
                question_id = step.get("extra_info", {}).get("question_id", "")
                
                if not ground_truth or not question_id:
                    continue
                
                # 提取 tool_calls
                tool_calls = extract_tool_calls(ground_truth)
                
                # 对每个 tool_call 调用 reward 服务
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_arguments = tool_call["arguments"]
                    
                    if dry_run:
                        # 干运行模式：使用模拟的 reward 值
                        reward = 0.5  # 模拟值
                    else:
                        # 调用 reward 服务
                        reward = call_reward_service(
                            reward_url,
                            question_id,
                            tool_name,
                            tool_arguments
                        )
                    
                    if reward is not None:
                        tool_rewards[tool_name].append(reward)
        
        except Exception as e:
            print(f"Warning: Error processing line: {e}", file=sys.stderr)
            continue
    
    return dict(tool_rewards)


def compute_statistics(tool_rewards: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    计算统计信息
    """
    statistics = {}
    
    for tool_name, rewards in tool_rewards.items():
        if rewards:
            statistics[tool_name] = {
                "count": len(rewards),
                "mean": sum(rewards) / len(rewards),
                "min": min(rewards),
                "max": max(rewards),
            }
    
    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="统计不同工具调用的平均 reward 分数"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="包含 rollout 文件的目录"
    )
    parser.add_argument(
        "--reward_url",
        type=str,
        default="http://localhost:5633",
        help="Reward 服务的 URL"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出统计结果到 JSON 文件（可选）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示进度条"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式：不调用 reward 服务，仅统计 tool call 数量"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("⚠️  Running in DRY-RUN mode: reward service will NOT be called")
        print("   Tool calls will be counted but rewards will be simulated (0.5)")
        print()
    
    # 指定要统计的文件
    rollout_files = [
        os.path.join(args.data_dir, "rollout_two_stage_gpt-5-2025-08-07-GlobalStandard.jsonl"),
        os.path.join(args.data_dir, "rollout_two_stage_qwen3-4b-thinking-2507_lora64.jsonl"),
        os.path.join(args.data_dir, "rollout_two_stage_qwen3-4b-thinking-2507_lora64_prm_orm.jsonl"),
        os.path.join(args.data_dir, "rollout_two_stage_qwen3-4b-thinking-2507_lora64_orm.jsonl"),
    ]
    
    # 检查文件是否存在
    existing_files = []
    for filepath in rollout_files:
        if os.path.isfile(filepath):
            existing_files.append(filepath)
        else:
            print(f"Warning: File not found: {filepath}", file=sys.stderr)
    
    rollout_files = existing_files
    
    if not rollout_files:
        print(f"Error: No rollout files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(rollout_files)} rollout files:")
    for f in rollout_files:
        print(f"  - {f}")
    print()
    
    # 处理每个文件并聚合结果
    all_tool_rewards = defaultdict(list)
    
    for filepath in rollout_files:
        print(f"\nProcessing: {filepath}")
        tool_rewards = process_rollout_file(
            filepath, 
            args.reward_url, 
            args.verbose,
            dry_run=args.dry_run
        )
        
        # 聚合结果
        for tool_name, rewards in tool_rewards.items():
            all_tool_rewards[tool_name].extend(rewards)
        
        # 打印单个文件的统计
        stats = compute_statistics(tool_rewards)
        print(f"\nStatistics for {os.path.basename(filepath)}:")
        for tool_name in sorted(stats.keys()):
            stat = stats[tool_name]
            print(f"  {tool_name:30s} - Count: {stat['count']:5d}, Mean: {stat['mean']:.4f}, Min: {stat['min']:.4f}, Max: {stat['max']:.4f}")
    
    # 计算并打印总体统计
    print("\n" + "=" * 80)
    print("Overall Statistics Across All Files:")
    print("=" * 80)
    overall_stats = compute_statistics(dict(all_tool_rewards))
    
    for tool_name in sorted(overall_stats.keys()):
        stat = overall_stats[tool_name]
        print(f"  {tool_name:30s} - Count: {stat['count']:5d}, Mean: {stat['mean']:.4f}, Min: {stat['min']:.4f}, Max: {stat['max']:.4f}")
    
    # 保存到文件
    if args.output:
        output_data = {
            "overall": overall_stats,
            "by_file": {},
            "metadata": {
                "dry_run": args.dry_run,
                "data_dir": args.data_dir,
                "reward_url": args.reward_url,
            }
        }
        
        for filepath in rollout_files:
            tool_rewards = process_rollout_file(
                filepath, 
                args.reward_url, 
                verbose=False,
                dry_run=args.dry_run
            )
            stats = compute_statistics(tool_rewards)
            output_data["by_file"][os.path.basename(filepath)] = stats
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
