
# # # 从文件夹移动出来
# import os
# import shutil

# def move_cat_files(src_dir, dst_dir):
#     # 创建目标文件夹如果不存在
#     if not os.path.exists(dst_dir):
#         os.makedirs(dst_dir)
    
#     # 用于跟踪目标文件夹中的文件编号
#     file_counter = 1
    
#     # 遍历源文件夹中的所有子文件夹
#     for root, dirs, files in os.walk(src_dir):
#         # 检查当前文件夹路径是否包含"2k"或者名为"images2k_orignal"
#         if "2k" in root or "images2k_orignal" in root:
#             print(f"跳过2k图像文件夹: {root}")
#             continue
            
#         for file in files:
#             if file.endswith(".fit"):
#                 # 构建源文件的完整路径
#                 src_file = os.path.join(root, file)
                
#                 # 构建新的文件名 (image1.fit, image2.fit, ...)
#                 new_filename = f"image{file_counter}.fit"
#                 dst_file = os.path.join(dst_dir, new_filename)
                
#                 # 如果目标文件已存在，增加计数器直到找到未使用的文件名
#                 while os.path.exists(dst_file):
#                     file_counter += 1
#                     new_filename = f"image{file_counter}.fit"
#                     dst_file = os.path.join(dst_dir, new_filename)
                
#                 # 复制文件
#                 shutil.copy(src_file, dst_file)
#                 print(f"已复制: {src_file} 到 {dst_file}")
                
#                 # 更新计数器
#                 file_counter += 1

# # 指定源文件夹和目标文件夹
# source_directory = '/home/flh/datasets/gs_imgs/tmp/gs_imgs_backup'
# destination_directory = '/home/flh/datasets/gs_imgs/quarter'
# move_cat_files(source_directory, destination_directory)






# # 从文件夹中随机抽取n个

import os
import random
import shutil

def move_random_files(src_dir, dst_dir, num_files):
    # 获取所有 .fit 文件
    all_files = [f for f in os.listdir(src_dir) if f.endswith(".fit")]
    
    # 随机选择 num_files 个文件
    selected_files = random.sample(all_files, min(num_files, len(all_files)))
    
    # 移动文件到目标文件夹
    for file in selected_files:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)
        shutil.move(src_file, dst_file)
        print(f"Moved: {src_file} to {dst_file}")

# 使用示例
source_directory = '/home/flh/datasets/gs_imgs/quarter'
destination_directory = '/home/flh/datasets/gs_imgs/data/test'
num_files = int(len(os.listdir(source_directory)))
move_random_files(source_directory, destination_directory, num_files)






# import os
# import random
# import shutil

# def move_random_files(src_dir, dst_dir, extra_dir,num_files=2000):
#     # 获取目标文件夹中已经存在的文件
#     existing_files = set(os.listdir(extra_dir))
    
#     # 获取所有 .fit 文件
#     all_files = [f for f in os.listdir(src_dir) if f.endswith(".fit")]
    
#     # 从源文件夹中过滤掉已经存在于目标文件夹中的文件
#     files_to_move = [f for f in all_files if f not in existing_files]
    
#     # 随机选择 num_files 个文件
#     selected_files = random.sample(files_to_move, min(num_files, len(files_to_move)))
    
#     # 移动文件到目标文件夹
#     for file in selected_files:
#         src_file = os.path.join(src_dir, file)
#         dst_file = os.path.join(dst_dir, file)
#         shutil.move(src_file, dst_file)
#         print(f"Moved: {src_file} to {dst_file}")

# # 使用示例
# source_directory = '/home/flh/datasets/gs_imgs/quarter'
# destination_directory = '/home/flh/datasets/gs_imgs/data/trainv2'
# extra_dir = '/home/flh/datasets/gs_imgs/data/train'
# move_random_files(source_directory, destination_directory, extra_dir, num_files=2000)

