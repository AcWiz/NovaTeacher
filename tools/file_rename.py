import os

# 指定文件夹路径
folder_path = '/home/flh/projects/datasets/Industrial/annos'

# 获取文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # 判断是否包含 '.jpg.txt'
        # 构建新的文件名
        new_filename = filename.replace('gt_', '').replace('.jpg', '')
        
        # 获取文件的完整路径
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")
