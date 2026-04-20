#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件恒星数量区间统计工具
统计文件夹中不同恒星数量区间的文件数量分布
"""

import os
import sys
import glob
from collections import defaultdict

def read_star_count_from_file(file_path):
    """
    从注释文件中读取恒星数量（第一行）
    
    Args:
        file_path (str): 注释文件路径
    
    Returns:
        int: 恒星数量，读取失败返回-1
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                return int(first_line)
            else:
                print(f"警告: 文件 {os.path.basename(file_path)} 为空")
                return -1
    except ValueError:
        print(f"警告: 文件 {os.path.basename(file_path)} 第一行不是有效数字: {first_line}")
        return -1
    except Exception as e:
        print(f"错误: 读取文件 {os.path.basename(file_path)} 时发生异常: {e}")
        return -1

def get_txt_files(input_path):
    """
    获取所有需要处理的txt文件
    
    Args:
        input_path (str): 输入路径（文件或文件夹）
    
    Returns:
        list: txt文件路径列表
    """
    txt_files = []
    
    if os.path.isfile(input_path):
        # 如果是单个文件
        if input_path.endswith('.txt'):
            txt_files.append(input_path)
        else:
            print(f"警告: {input_path} 不是txt文件")
    elif os.path.isdir(input_path):
        # 如果是文件夹，查找所有txt文件
        pattern = os.path.join(input_path, "*.txt")
        txt_files = glob.glob(pattern)
        
        # 也查找子文件夹中的txt文件
        pattern_recursive = os.path.join(input_path, "**", "*.txt")
        txt_files.extend(glob.glob(pattern_recursive, recursive=True))
        
        # 去重并排序
        txt_files = list(set(txt_files))
        txt_files.sort()
    else:
        print(f"错误: 路径 {input_path} 不存在")
    
    return txt_files

def categorize_files_by_star_count(file_star_counts):
    """
    根据恒星数量将文件分类到不同区间
    
    Args:
        file_star_counts (dict): {文件路径: 恒星数量}
    
    Returns:
        dict: 各区间的文件统计 {区间名称: [文件列表]}
    """
    # 定义区间
    intervals = [
        (0, 10, "(0,10]"),
        (10, 20, "(10,20]"),
        (20, 30, "(20,30]"),
        (30, 40, "(30,40]"),
        (40, 50, "(40,50]"),
        (50, 60, "(50,60]"),
        (60, 70, "(60,70]"),
        (70, 80, "(70,80]"),
        (80, 90, "(80,90]"),
        (90, 100, "(90,100]"),
        (100, float('inf'), ">100")
    ]
    
    # 初始化分类结果
    categorized_files = {interval[2]: [] for interval in intervals}
    
    # 将每个文件分类到对应区间
    for file_path, star_count in file_star_counts.items():
        if star_count < 0:  # 跳过读取失败的文件
            continue
            
        for start, end, label in intervals:
            if start < star_count <= end:
                categorized_files[label].append((file_path, star_count))
                break
    
    return categorized_files

def print_statistics(categorized_files, total_files, invalid_files):
    """
    打印统计结果
    
    Args:
        categorized_files (dict): 各区间的文件分类
        total_files (int): 总文件数量
        invalid_files (int): 无效文件数量
    """
    print("\n" + "="*80)
    print("文件恒星数量区间统计结果")
    print("="*80)
    
    valid_files = total_files - invalid_files
    
    print(f"总文件数: {total_files}")
    print(f"有效文件数: {valid_files}")
    if invalid_files > 0:
        print(f"无效文件数: {invalid_files}")
    print("-"*80)
    
    for interval, files in categorized_files.items():
        file_count = len(files)
        percentage = (file_count / valid_files * 100) if valid_files > 0 else 0
        
        # 简单的条形图
        bar_length = int(percentage / 2) if percentage <= 100 else 50
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"{interval:<10}: {file_count:>6} 个文件 ({percentage:>5.1f}%) {bar}")
    
    print("-"*80)
    print(f"{'总计':<10}: {valid_files:>6} 个文件 (100.0%)")

def print_detailed_file_list(categorized_files, show_details=False):
    """
    打印详细的文件列表
    
    Args:
        categorized_files (dict): 各区间的文件分类
        show_details (bool): 是否显示详细信息
    """
    if not show_details:
        return
        
    print("\n" + "="*80)
    print("详细文件列表")
    print("="*80)
    
    for interval, files in categorized_files.items():
        if files:
            print(f"\n{interval} 区间 ({len(files)} 个文件):")
            print("-" * 40)
            for file_path, star_count in sorted(files, key=lambda x: x[1]):
                filename = os.path.basename(file_path)
                print(f"  {filename:<30} ({star_count:>3} 颗恒星)")

def save_statistics_to_file(categorized_files, file_info, output_file="file_star_statistics.txt"):
    """
    将统计结果保存到文件
    
    Args:
        categorized_files (dict): 各区间的文件分类
        file_info (dict): 文件处理信息
        output_file (str): 输出文件名
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("文件恒星数量区间统计结果\n")
            f.write("="*80 + "\n")
            f.write(f"处理路径: {file_info['input_path']}\n")
            f.write(f"总文件数: {file_info['total_files']}\n")
            f.write(f"有效文件数: {file_info['valid_files']}\n")
            if file_info['invalid_files'] > 0:
                f.write(f"无效文件数: {file_info['invalid_files']}\n")
            f.write("="*80 + "\n\n")
            
            # 统计摘要
            for interval, files in categorized_files.items():
                file_count = len(files)
                percentage = (file_count / file_info['valid_files'] * 100) if file_info['valid_files'] > 0 else 0
                f.write(f"{interval:<10}: {file_count:>6} 个文件 ({percentage:>5.1f}%)\n")
            
            f.write("-"*80 + "\n")
            f.write(f"{'总计':<10}: {file_info['valid_files']:>6} 个文件 (100.0%)\n")
            
            # 详细文件列表
            f.write("\n\n详细文件列表:\n")
            f.write("="*80 + "\n")
            
            for interval, files in categorized_files.items():
                if files:
                    f.write(f"\n{interval} 区间 ({len(files)} 个文件):\n")
                    f.write("-" * 40 + "\n")
                    for file_path, star_count in sorted(files, key=lambda x: x[1]):
                        filename = os.path.basename(file_path)
                        f.write(f"  {filename:<40} ({star_count:>3} 颗恒星) - {file_path}\n")
        
        print(f"\n统计结果已保存到: {output_file}")
    
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python star_statistics.py <文件或文件夹路径> [输出文件名] [--详情]")
        print("示例: python star_statistics.py /path/to/folder/")
        print("      python star_statistics.py /path/to/folder/ result.txt")
        print("      python star_statistics.py /path/to/folder/ result.txt --详情")
        return
    
    input_path = sys.argv[1]
    output_file = "file_star_statistics.txt"
    show_details = False
    
    # 解析参数
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg == "--详情" or arg == "--details":
                show_details = True
            elif not arg.startswith("--"):
                output_file = arg
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误: 路径 '{input_path}' 不存在")
        return
    
    print(f"正在处理路径: {input_path}")
    
    # 获取所有txt文件
    txt_files = get_txt_files(input_path)
    
    if not txt_files:
        print("错误: 没有找到任何txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    # 读取每个文件的恒星数量
    file_star_counts = {}
    invalid_files = 0
    
    for i, file_path in enumerate(txt_files, 1):
        print(f"处理文件 {i}/{len(txt_files)}: {os.path.basename(file_path)}")
        
        star_count = read_star_count_from_file(file_path)
        
        if star_count >= 0:
            file_star_counts[file_path] = star_count
        else:
            invalid_files += 1
    
    if not file_star_counts:
        print("错误: 没有有效的文件数据")
        return
    
    print(f"\n数据处理完成:")
    print(f"- 总文件数: {len(txt_files)}")
    print(f"- 有效文件数: {len(file_star_counts)}")
    if invalid_files > 0:
        print(f"- 无效文件数: {invalid_files}")
    
    # 按恒星数量区间分类文件
    categorized_files = categorize_files_by_star_count(file_star_counts)
    
    # 打印统计结果
    print_statistics(categorized_files, len(txt_files), invalid_files)
    
    # 打印详细文件列表（如果需要）
    print_detailed_file_list(categorized_files, show_details)
    
    # 保存结果
    file_info = {
        'input_path': input_path,
        'total_files': len(txt_files),
        'valid_files': len(file_star_counts),
        'invalid_files': invalid_files
    }
    save_statistics_to_file(categorized_files, file_info, output_file)

if __name__ == "__main__":
    main()