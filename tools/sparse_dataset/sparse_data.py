# 按照每张图片进行稀疏
import os
import random

def count_total_targets(dataset_path):
    total_targets = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    total_targets += len(f.readlines())
    return total_targets

def process_dataset(dataset_path, output_path, sparse_ratio=0.5):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    total_targets = count_total_targets(dataset_path)
    targets_to_keep = int(total_targets * sparse_ratio)
    
    # 遍历每个文件进行稀疏处理
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                
                # 读取文件中的目标
                with open(file_path, 'r') as f:
                    lines = [line for line in f if line.strip()]  # 排除空行
                total_targets_in_file = len(lines)
                
                # 计算需要保留的目标数
                targets_to_keep_in_file = int(total_targets_in_file * sparse_ratio)
                
                # 随机选择保留的目标
                kept_indices = random.sample(range(total_targets_in_file), targets_to_keep_in_file)
                kept_targets = [lines[i] for i in kept_indices]

                # 创建新的稀疏文件并保存
                relative_path = os.path.relpath(file_path, dataset_path)
                new_file_path = os.path.join(output_path, relative_path)
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                
                with open(new_file_path, 'w') as f:
                    f.writelines(kept_targets)
    
    print(f"原始目标总数: {total_targets}")
    print(f"保留的目标数: {targets_to_keep}")
    print(f"稀疏处理完成，新文件已创建在 {output_path} 文件夹中。")

# 使用示例
dataset_path = 'data/CSST_data/split_data/train/annfiles'  # 请替换为您的数据集路径
output_path = 'data/CSST_data/split_data/train/sparse_50'  # 请替换为您想要保存稀疏数据集的路径
process_dataset(dataset_path, output_path, sparse_ratio=0.5)








# # 数据集整体稀疏
# import os
# import random
# import shutil

# def count_total_targets(dataset_path):
#     total_targets = 0
#     for root, dirs, files in os.walk(dataset_path):
#         for file in files:
#             if file.endswith('.txt'):
#                 with open(os.path.join(root, file), 'r') as f:
#                     total_targets += len(f.readlines())
#     return total_targets

# def process_dataset(dataset_path, output_path, sparse_ratio=0.7):
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
    
#     total_targets = count_total_targets(dataset_path)
#     targets_to_keep = int(total_targets * sparse_ratio)
    
#     all_targets = []
#     file_paths = []

#     # 收集所有目标和文件路径
#     for root, dirs, files in os.walk(dataset_path):
#         for file in files:
#             if file.endswith('.txt'):
#                 file_path = os.path.join(root, file)
#                 with open(file_path, 'r') as f:
#                     lines = f.readlines()
#                     all_targets.extend(lines)
#                     file_paths.extend([file_path] * len(lines))

#     # 随机选择要保留的目标
#     kept_indices = random.sample(range(len(all_targets)), targets_to_keep)
#     kept_targets = [all_targets[i] for i in kept_indices]
#     kept_file_paths = [file_paths[i] for i in kept_indices]

#     # 创建新的稀疏文件
#     for file_path in set(file_paths):
#         relative_path = os.path.relpath(file_path, dataset_path)
#         new_file_path = os.path.join(output_path, relative_path)
#         os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        
#         with open(new_file_path, 'w') as f:
#             for target, path in zip(kept_targets, kept_file_paths):
#                 if path == file_path:
#                     f.write(target)
    
#     print(f"原始目标总数: {total_targets}")
#     print(f"保留的目标数: {targets_to_keep}")
#     print(f"稀疏处理完成，新文件已创建在 {output_path} 文件夹中。")

# # 使用示例
# dataset_path = ''  # 请替换为您的数据集路径
# output_path = 'data/SparseDet/new_ged/sparse_30'  # 请替换为您想要保存稀疏数据集的路径
# process_dataset(dataset_path, output_path)