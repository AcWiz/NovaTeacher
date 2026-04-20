# 按照数据集为整体进行稀疏

import os
import random

def collect_all_targets(dataset_path):
    """收集数据集中所有的目标及其所属文件"""
    all_targets = []
    file_paths = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = [line for line in f if line.strip()]  # 排除空行
                
                for line in lines:
                    all_targets.append(line)
                    file_paths.append(file_path)
    
    return all_targets, file_paths

def process_dataset(dataset_path, output_path, sparse_ratio=0.7):
    """对整个数据集进行稀疏处理"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 收集所有目标及其所属文件
    all_targets, file_paths = collect_all_targets(dataset_path)
    total_targets = len(all_targets)
    
    # 计算需要保留的目标数
    targets_to_keep = int(total_targets * sparse_ratio)
    
    # 随机选择保留的目标索引
    kept_indices = random.sample(range(total_targets), targets_to_keep)
    
    # 创建一个字典，用于存储每个文件需要保留的目标
    file_to_targets = {}
    for idx in kept_indices:
        target = all_targets[idx]
        file_path = file_paths[idx]
        
        if file_path not in file_to_targets:
            file_to_targets[file_path] = []
        
        file_to_targets[file_path].append(target)
    
    # 创建新的稀疏文件并保存
    for file_path, targets in file_to_targets.items():
        relative_path = os.path.relpath(file_path, dataset_path)
        new_file_path = os.path.join(output_path, relative_path)
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        
        with open(new_file_path, 'w') as f:
            f.writelines(targets)
    
    # 为原数据集中的空文件创建空文件
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, dataset_path)
                new_file_path = os.path.join(output_path, relative_path)
                
                # 如果该文件在稀疏处理后没有目标，则创建一个空文件
                if file_path not in file_to_targets:
                    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                    open(new_file_path, 'w').close()
    
    print(f"原始目标总数: {total_targets}")
    print(f"保留的目标数: {targets_to_keep}")
    print(f"稀疏率: {sparse_ratio * 100:.1f}%")
    print(f"稀疏处理完成，新文件已创建在 {output_path} 文件夹中。")

# 使用示例
dataset_path = '/home/flh/projects/datasets/PCB/train/annos'  # 请替换为您的数据集路径
output_path = '/home/flh/projects/datasets/PCB/train/sparse/sparse_50'  # 请替换为您想要保存稀疏数据集的路径
process_dataset(dataset_path, output_path, sparse_ratio=0.5)