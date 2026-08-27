#!/usr/bin/env python3
"""
统计数据分布脚本
分析product_examples.jsonl和test_dataset_examples.jsonl中的数据分布

生成的图表：
1. output/category_distribution.pdf - 产品类目分布直方图（Top 20）
2. output/wanted_features_distribution.pdf - 不同question_type的wanted_features数量分布箱线图
3. output/add_on_deals_statistics.pdf - add_on_deals的商品数量和wanted_features总数分布（双图）
4. output/dialogue_tokens_distribution.pdf - dialogue的tokens数分布直方图
5. output/haystack_distributions.pdf - haystack_sessions的长度和tokens总数分布直方图（双图）

使用方法：
    python3 src/basic_statistics.py

注意：
- 使用tiktoken精确计算token数量（cl100k_base编码）
- 所有图表保存在output目录下
- 统计摘要会打印到控制台
- 对于add_on_deals类型，统计每个样本的商品数量（preferences数量）和wanted_features总数
"""

import json
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不需要显示
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import tiktoken

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 初始化tiktoken编码器
enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """
    使用tiktoken精确计算文本的token数量
    """
    if isinstance(text, str):
        return len(enc.encode(text, disallowed_special=()))
    elif isinstance(text, list):
        # 如果是对话列表，计算所有消息的token总数
        total = 0
        for msg in text:
            if isinstance(msg, dict) and 'content' in msg:
                total += len(enc.encode(msg['content'], disallowed_special=()))
        return total
    return 0


def load_jsonl(filepath):
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze_product_categories(data):
    """分析产品类目分布"""
    categories = []
    for item in data:
        if 'category' in item:
            # 取类目的第一级
            cat = item['category'].split(' - ')[0] if ' - ' in item['category'] else item['category']
            categories.append(cat)
    
    return Counter(categories)


def analyze_test_dataset(data):
    """分析测试数据集"""
    # 按question_type统计wanted_features数量
    features_by_type = defaultdict(list)
    
    # 统计dialogue的tokens数
    dialogue_tokens = []
    
    # 统计haystack_sessions的长度
    haystack_lengths = []
    
    # 统计haystack_sessions中所有session的tokens总数
    haystack_total_tokens = []
    
    # add_on_deals特殊统计
    add_on_deals_preferences_count = []  # 每个样本的商品数量（preferences数量）
    add_on_deals_total_features = []     # 每个样本的wanted_features总数
    
    for item in data:
        question_type = item.get('question_type', 'unknown')
        
        if 'answer' not in item:
            continue
            
        answer = item['answer']
        
        # 处理不同的数据结构
        if question_type == 'add_on_deals':
            # add_on_deals: wanted_features和dialogue在preferences数组中
            if 'preferences' in answer:
                # 统计商品数量（preferences数量）
                preferences_count = len(answer['preferences'])
                add_on_deals_preferences_count.append(preferences_count)
                
                # 统计所有preferences的wanted_features总数
                total_features = 0
                for pref in answer['preferences']:
                    # wanted_features数量
                    if 'wanted_features' in pref:
                        total_features += len(pref['wanted_features'])
                    
                    # dialogue tokens
                    if 'dialogue' in pref:
                        tokens = count_tokens(pref['dialogue'])
                        dialogue_tokens.append(tokens)
                
                # 记录该样本的wanted_features总数
                add_on_deals_total_features.append(total_features)
                features_by_type[question_type].append(total_features)
        else:
            # single_product等其他类型: wanted_features和dialogue直接在answer下
            # wanted_features数量
            if 'wanted_features' in answer:
                features_count = len(answer['wanted_features'])
                features_by_type[question_type].append(features_count)
            
            # dialogue tokens
            if 'dialogue' in answer:
                tokens = count_tokens(answer['dialogue'])
                dialogue_tokens.append(tokens)
        
        # haystack_sessions长度和tokens (所有类型都一样)
        if 'haystack_sessions' in item:
            haystack_sessions = item['haystack_sessions']
            haystack_lengths.append(len(haystack_sessions))
            
            # 计算所有session的tokens总数
            total_tokens = 0
            for session in haystack_sessions:
                if isinstance(session, list):
                    total_tokens += count_tokens(session)
            haystack_total_tokens.append(total_tokens)
    
    return {
        'features_by_type': features_by_type,
        'dialogue_tokens': dialogue_tokens,
        'haystack_lengths': haystack_lengths,
        'haystack_total_tokens': haystack_total_tokens,
        'add_on_deals_preferences_count': add_on_deals_preferences_count,
        'add_on_deals_total_features': add_on_deals_total_features
    }


def plot_category_distribution(category_counts, output_path='output/category_distribution.pdf'):
    """绘制产品类目分布直方图"""
    # 按数量排序，取前20个类目
    top_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = list(top_categories.keys())
    counts = list(top_categories.values())
    
    bars = ax.bar(range(len(categories)), counts, color='steelblue', edgecolor='navy', alpha=0.7)
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Product Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    # ax.set_title('Product Category Distribution (Top 20)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Category distribution plot saved to {output_path}")
    plt.close()


def plot_wanted_features_distribution(features_by_type, output_path='output/wanted_features_distribution.pdf'):
    """绘制wanted_features数量分布直方图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 为每个question_type创建数据
    all_data = []
    labels = []
    colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'mediumpurple']
    
    for i, (qtype, counts) in enumerate(sorted(features_by_type.items())):
        all_data.append(counts)
        labels.append(f"{qtype}\n(n={len(counts)})")
    
    # 创建箱线图
    bp = ax.boxplot(all_data, tick_labels=labels, patch_artist=True, widths=0.6,
                     showmeans=True, meanline=True)
    
    # 设置颜色
    for patch, color in zip(bp['boxes'], colors[:len(all_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # 添加统计信息
    for i, counts in enumerate(all_data, 1):
        mean_val = np.mean(counts)
        median_val = np.median(counts)
        ax.text(i, mean_val, f'μ={mean_val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Question Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Wanted Features', fontsize=12, fontweight='bold')
    # ax.set_title('Distribution of Wanted Features Count by Question Type', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Wanted features distribution plot saved to {output_path}")
    plt.close()


def plot_dialogue_tokens_distribution(dialogue_tokens, output_path='output/dialogue_tokens_distribution.pdf'):
    """绘制dialogue的tokens数分布直方图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建直方图
    n, bins, patches = ax.hist(dialogue_tokens, bins=30, color='steelblue', 
                                edgecolor='navy', alpha=0.7)
    
    # 渐变色效果
    cm = plt.cm.viridis
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # 添加统计信息
    mean_val = np.mean(dialogue_tokens)
    median_val = np.median(dialogue_tokens)
    std_val = np.std(dialogue_tokens)
    
    textstr = f'Mean: {mean_val:.0f}\nMedian: {median_val:.0f}\nStd: {std_val:.0f}\nSamples: {len(dialogue_tokens)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.75, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Number of Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    # ax.set_title('Distribution of Dialogue Token Count', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Dialogue tokens distribution plot saved to {output_path}")
    plt.close()


def plot_add_on_deals_statistics(preferences_count, total_features, 
                                   output_path='output/add_on_deals_statistics.pdf'):
    """绘制add_on_deals的商品数量和wanted_features总数分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：商品数量（preferences数量）分布
    n1, bins1, patches1 = ax1.hist(preferences_count, bins=range(min(preferences_count), 
                                                                   max(preferences_count) + 2), 
                                     color='mediumpurple', edgecolor='indigo', alpha=0.7, 
                                     align='left', rwidth=0.8)
    
    mean_pref = np.mean(preferences_count)
    median_pref = np.median(preferences_count)
    mode_pref = max(set(preferences_count), key=preferences_count.count)
    
    textstr1 = f'Mean: {mean_pref:.2f}\nMedian: {median_pref:.0f}\nMode: {mode_pref}\nSamples: {len(preferences_count)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.65, 0.97, textstr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # 添加数值标签
    for i, (count, height) in enumerate(zip(bins1[:-1], n1)):
        if height > 0:
            ax1.text(count, height, f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Number of Products (Preferences Count)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    # ax1.set_title('Add-on Deals: Products Count Distribution', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticks(range(min(preferences_count), max(preferences_count) + 1))
    
    # 右图：wanted_features总数分布
    n2, bins2, patches2 = ax2.hist(total_features, bins=20, color='teal', 
                                     edgecolor='darkslategray', alpha=0.7)
    
    # 渐变色效果
    cm = plt.cm.GnBu
    bin_centers = 0.5 * (bins2[:-1] + bins2[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col) if max(col) > 0 else 1
    for c, p in zip(col, patches2):
        plt.setp(p, 'facecolor', cm(c))
    
    mean_feat = np.mean(total_features)
    median_feat = np.median(total_features)
    std_feat = np.std(total_features)
    
    textstr2 = f'Mean: {mean_feat:.2f}\nMedian: {median_feat:.0f}\nStd: {std_feat:.2f}\nSamples: {len(total_features)}'
    ax2.text(0.65, 0.97, textstr2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    ax2.set_xlabel('Total Number of Wanted Features', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    # ax2.set_title('Add-on Deals: Total Wanted Features Distribution', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Add-on deals statistics plot saved to {output_path}")
    plt.close()


def plot_haystack_distributions(haystack_lengths, haystack_total_tokens, 
                                 output_path='output/haystack_distributions.pdf'):
    """绘制haystack_sessions的长度和tokens总数分布直方图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：haystack_sessions长度分布
    n1, bins1, patches1 = ax1.hist(haystack_lengths, bins=30, color='coral', 
                                     edgecolor='darkred', alpha=0.7)
    
    # 渐变色效果
    cm = plt.cm.Oranges
    bin_centers = 0.5 * (bins1[:-1] + bins1[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches1):
        plt.setp(p, 'facecolor', cm(c))
    
    mean_len = np.mean(haystack_lengths)
    median_len = np.median(haystack_lengths)
    std_len = np.std(haystack_lengths)
    
    textstr1 = f'Mean: {mean_len:.1f}\nMedian: {median_len:.1f}\nStd: {std_len:.1f}\nSamples: {len(haystack_lengths)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.65, 0.97, textstr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    ax1.set_xlabel('Number of Sessions', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    # ax1.set_title('Distribution of Haystack Sessions Length', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 右图：haystack_sessions tokens总数分布
    n2, bins2, patches2 = ax2.hist(haystack_total_tokens, bins=30, color='mediumseagreen', 
                                     edgecolor='darkgreen', alpha=0.7)
    
    # 渐变色效果
    cm = plt.cm.Greens
    bin_centers = 0.5 * (bins2[:-1] + bins2[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches2):
        plt.setp(p, 'facecolor', cm(c))
    
    mean_tokens = np.mean(haystack_total_tokens)
    median_tokens = np.median(haystack_total_tokens)
    std_tokens = np.std(haystack_total_tokens)
    
    textstr2 = f'Mean: {mean_tokens:.0f}\nMedian: {median_tokens:.0f}\nStd: {std_tokens:.0f}\nSamples: {len(haystack_total_tokens)}'
    ax2.text(0.65, 0.97, textstr2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    ax2.set_xlabel('Total Number of Tokens', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    # ax2.set_title('Distribution of Haystack Sessions Total Tokens', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Haystack distributions plot saved to {output_path}")
    plt.close()


def print_statistics(product_data, test_data):
    """打印统计信息"""
    print("\n" + "="*60)
    print("STATISTICS SUMMARY")
    print("="*60)
    
    # Product statistics
    print("\n📦 Product Statistics:")
    print(f"   Total products: {len(product_data)}")
    categories = analyze_product_categories(product_data)
    print(f"   Unique categories: {len(categories)}")
    print(f"   Top 5 categories:")
    for cat, count in categories.most_common(5):
        print(f"      - {cat}: {count}")
    
    # Test dataset statistics
    print("\n📊 Test Dataset Statistics:")
    print(f"   Total samples: {len(test_data)}")
    
    stats = analyze_test_dataset(test_data)
    
    print(f"\n   Question Types:")
    for qtype, features in stats['features_by_type'].items():
        print(f"      - {qtype}: {len(features)} samples")
        print(f"        Average wanted features: {np.mean(features):.2f}")
    
    # Add-on Deals 特殊统计
    if stats['add_on_deals_preferences_count']:
        print(f"\n   🛍️ Add-on Deals Statistics:")
        print(f"      Total samples: {len(stats['add_on_deals_preferences_count'])}")
        print(f"\n      Products Count (Preferences):")
        print(f"         Mean: {np.mean(stats['add_on_deals_preferences_count']):.2f}")
        print(f"         Median: {np.median(stats['add_on_deals_preferences_count']):.0f}")
        print(f"         Min: {np.min(stats['add_on_deals_preferences_count'])}")
        print(f"         Max: {np.max(stats['add_on_deals_preferences_count'])}")
        print(f"\n      Total Wanted Features:")
        print(f"         Mean: {np.mean(stats['add_on_deals_total_features']):.2f}")
        print(f"         Median: {np.median(stats['add_on_deals_total_features']):.0f}")
        print(f"         Min: {np.min(stats['add_on_deals_total_features'])}")
        print(f"         Max: {np.max(stats['add_on_deals_total_features'])}")
    
    print(f"\n   Dialogue Tokens:")
    print(f"      Mean: {np.mean(stats['dialogue_tokens']):.0f}")
    print(f"      Median: {np.median(stats['dialogue_tokens']):.0f}")
    print(f"      Min: {np.min(stats['dialogue_tokens'])}")
    print(f"      Max: {np.max(stats['dialogue_tokens'])}")
    
    print(f"\n   Haystack Sessions Length:")
    print(f"      Mean: {np.mean(stats['haystack_lengths']):.1f}")
    print(f"      Median: {np.median(stats['haystack_lengths']):.0f}")
    print(f"      Min: {np.min(stats['haystack_lengths'])}")
    print(f"      Max: {np.max(stats['haystack_lengths'])}")
    
    print(f"\n   Haystack Total Tokens:")
    print(f"      Mean: {np.mean(stats['haystack_total_tokens']):.0f}")
    print(f"      Median: {np.median(stats['haystack_total_tokens']):.0f}")
    print(f"      Min: {np.min(stats['haystack_total_tokens'])}")
    print(f"      Max: {np.max(stats['haystack_total_tokens'])}")
    
    print("\n" + "="*60)


def main():
    """主函数"""
    import os
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    print("Loading data...")
    
    # 加载数据
    product_data = load_jsonl('data/products.jsonl')
    test_data = load_jsonl('data/long_term_conversations.jsonl')
    
    print(f"✓ Loaded {len(product_data)} products")
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # 分析产品类目分布
    print("\nAnalyzing product categories...")
    category_counts = analyze_product_categories(product_data)
    plot_category_distribution(category_counts)
    
    # 分析测试数据集
    print("\nAnalyzing test dataset...")
    test_stats = analyze_test_dataset(test_data)
    
    # 绘制wanted_features分布
    print("\nPlotting wanted features distribution...")
    plot_wanted_features_distribution(test_stats['features_by_type'])
    
    # 绘制add_on_deals统计
    if test_stats['add_on_deals_preferences_count']:
        print("\nPlotting add-on deals statistics...")
        plot_add_on_deals_statistics(test_stats['add_on_deals_preferences_count'],
                                       test_stats['add_on_deals_total_features'])
    
    # 绘制dialogue tokens分布
    print("\nPlotting dialogue tokens distribution...")
    plot_dialogue_tokens_distribution(test_stats['dialogue_tokens'])
    
    # 绘制haystack分布
    print("\nPlotting haystack distributions...")
    plot_haystack_distributions(test_stats['haystack_lengths'], 
                                 test_stats['haystack_total_tokens'])
    
    # 打印统计信息
    print_statistics(product_data, test_data)
    
    print("\n✅ All plots generated successfully!")
    print("   Check the 'output' folder for the generated charts.")


if __name__ == '__main__':
    main()
