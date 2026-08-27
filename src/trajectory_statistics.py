#!/usr/bin/env python3
"""
统计脚本：分析rollout文件中的统计信息
- 总Steps数量
- 总tool calls数量
- 最后一个Step的prompt和ground_truth的tokens数量
"""

import os
import re
import json
from typing import List, Dict, Any

import tiktoken


def extract_tool_calls(content: str) -> List[str]:
    """
    从ground_truth内容中提取有效的tool calls
    逻辑参考agent_loop.py中的act方法
    """
    tool_calls = []
    
    # 提取所有<tool_call>...</tool_call>块
    tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    
    for tool_call_block in tool_call_blocks:
        tool_call_block = tool_call_block.strip()
        for tool_call in tool_call_block.split("\n"):
            tool_call = tool_call.strip()
            if not tool_call:
                continue
            
            # 验证是否为有效的JSON格式
            try:
                jsonobj = json.loads(tool_call)
                name = jsonobj.get("name")
                arguments = jsonobj.get("arguments")
                
                # 验证是否有name和arguments字段
                if name and arguments and isinstance(arguments, dict):
                    tool_calls.append(tool_call)
            except Exception:
                # 无效的tool call，跳过
                continue
    
    return tool_calls


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    使用tiktoken统计tokens数量
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def count_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    """
    统计prompt中所有content的tokens数量
    """
    total_tokens = 0
    for message in messages:
        content = message.get("content", "")
        total_tokens += count_tokens(content)
    return total_tokens


def analyze_file(filepath: str) -> Dict[str, Any]:
    """
    分析单个JSONL文件
    """
    num_trajectories = 0
    total_steps = 0
    total_tool_calls = 0
    total_last_step_prompt_tokens = 0
    total_last_step_ground_truth_tokens = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 每行是一个JSON数组，代表一个trajectory
            trajectory = json.loads(line)
            
            if not trajectory or not isinstance(trajectory, list):
                continue
            
            # 统计轨迹数量
            num_trajectories += 1
            
            # 统计该trajectory的steps数量
            trajectory_steps = len(trajectory)
            total_steps += trajectory_steps
            
            # 统计该trajectory的tool calls数量
            trajectory_tool_calls = 0
            for step in trajectory:
                ground_truth = step.get("reward_model", {}).get("ground_truth", "")
                tool_calls = extract_tool_calls(ground_truth)
                trajectory_tool_calls += len(tool_calls)
            total_tool_calls += trajectory_tool_calls
            
            # 获取最后一个step的信息
            if trajectory_steps > 0:
                last_step = trajectory[-1]
                
                # 统计prompt的tokens
                prompt = last_step.get("prompt", [])
                total_last_step_prompt_tokens += count_prompt_tokens(prompt)
                
                # 统计ground_truth的tokens
                ground_truth = last_step.get("reward_model", {}).get("ground_truth", "")
                total_last_step_ground_truth_tokens += count_tokens(ground_truth)
    
    return {
        "num_trajectories": num_trajectories,
        "total_steps": total_steps,
        "total_tool_calls": total_tool_calls,
        "total_last_step_prompt_tokens": total_last_step_prompt_tokens,
        "total_last_step_ground_truth_tokens": total_last_step_ground_truth_tokens,
    }


def main():
    """
    主函数：统计两个文件的信息
    """
    files = [
        "data/rollout_two_stage_gpt-5-2025-08-07-GlobalStandard.jsonl",
        "data/rollout_two_stage_qwen3-4b-thinking-2507_lora64.jsonl",
        "data/rollout_two_stage_qwen3-4b-thinking-2507_lora64_prm_orm.jsonl",
        "data/rollout_two_stage_qwen3-4b-thinking-2507_lora64_orm.jsonl",
    ]
    
    print("=" * 100)
    print("Trajectory Statistics")
    print("=" * 100)
    print()
    
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"  ❌ 文件不存在: {filepath}")
            continue

        print(f"📊 分析文件: {filepath}")
        print("-" * 100)
        
        try:
            stats = analyze_file(filepath)
            
            num_traj = stats['num_trajectories']
            total_steps = stats['total_steps']
            total_tool_calls = stats['total_tool_calls']
            total_prompt_tokens = stats['total_last_step_prompt_tokens']
            total_gt_tokens = stats['total_last_step_ground_truth_tokens']
            
            print(f"  轨迹数量:                            {num_traj:,}")
            print()
            
            # (1) Steps统计
            print(f"  总Steps数量:                         {total_steps:,}")
            if num_traj > 0:
                avg_steps = total_steps / num_traj
                print(f"  每个轨迹的平均Steps数量:             {avg_steps:.2f}")
            print()
            
            # (2) Tool Calls统计
            print(f"  总Tool Calls数量:                    {total_tool_calls:,}")
            if num_traj > 0:
                avg_tool_calls_per_traj = total_tool_calls / num_traj
                print(f"  每个轨迹的平均Tool Calls数量:        {avg_tool_calls_per_traj:.2f}")
            print()
            
            # (3) 最后一个Step的Tokens统计
            print(f"  所有最后一个Step的Prompt总tokens:    {total_prompt_tokens:,}")
            if num_traj > 0:
                avg_prompt_tokens = total_prompt_tokens / num_traj
                print(f"  每个轨迹最后一个Step的平均Prompt tokens: {avg_prompt_tokens:.2f}")
            print()
            
            print(f"  所有最后一个Step的Ground Truth总tokens: {total_gt_tokens:,}")
            if num_traj > 0:
                avg_gt_tokens = total_gt_tokens / num_traj
                print(f"  每个轨迹最后一个Step的平均Ground Truth tokens: {avg_gt_tokens:.2f}")
            
            print()
            print()
            
        except FileNotFoundError:
            print(f"  ❌ 文件不存在: {filepath}")
            print()
        except Exception as e:
            print(f"  ❌ 处理文件时出错: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 100)


if __name__ == "__main__":
    main()
