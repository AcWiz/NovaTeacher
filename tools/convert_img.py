
#!/usr/bin/env python3
"""
图片格式转换工具
将指定文件夹中的所有图片转换为 JPG 格式，保持原文件名
用法
基本用法（保留原文件）：
python convert_images_to_jpg.py /path/to/your/images
删除原文件（转换成功后删除原始文件）：
python convert_images_to_jpg.py /path/to/your/images --delete-original
"""

import os
from PIL import Image
from pathlib import Path

def convert_images_to_jpg(folder_path, delete_original=False):
    """
    将文件夹中的所有图片转换为 JPG 格式
    
    参数:
        folder_path: 图片文件夹路径
        delete_original: 是否删除原始文件（默认为 False）
    """
    # 支持的图片格式
    supported_formats = {'.png', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.ico', '.jpeg'}
    
    # 转换计数
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    # 获取文件夹路径对象
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return
    
    if not folder.is_dir():
        print(f"错误: 路径不是文件夹: {folder_path}")
        return
    
    print(f"开始处理文件夹: {folder_path}")
    print("-" * 50)
    
    # 遍历文件夹中的所有文件
    for file_path in folder.iterdir():
        # 跳过文件夹
        if file_path.is_dir():
            continue
        
        # 获取文件扩展名（小写）
        file_ext = file_path.suffix.lower()
        
        # 跳过已经是 JPG 格式的文件
        if file_ext == '.jpg':
            skipped_count += 1
            print(f"跳过 (已是JPG): {file_path.name}")
            continue
        
        # 检查是否是支持的图片格式
        if file_ext not in supported_formats:
            skipped_count += 1
            continue
        
        try:
            # 打开图片
            img = Image.open(file_path)
            
            # 如果图片有透明通道(RGBA 或 LA)，转换为 RGB
            if img.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                # 如果有 alpha 通道，使用它进行合成
                if img.mode == 'RGBA' or img.mode == 'LA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 生成新的文件名（保持原文件名，只改扩展名）
            new_file_path = file_path.with_suffix('.jpg')
            
            # 保存为 JPG 格式
            img.save(new_file_path, 'JPEG', quality=95)
            
            print(f"✓ 转换成功: {file_path.name} -> {new_file_path.name}")
            converted_count += 1
            
            # 如果需要删除原始文件
            if delete_original and file_path != new_file_path:
                file_path.unlink()
                print(f"  已删除原文件: {file_path.name}")
            
        except Exception as e:
            print(f"✗ 转换失败: {file_path.name} - 错误: {str(e)}")
            error_count += 1
    
    # 打印统计信息
    print("-" * 50)
    print(f"处理完成!")
    print(f"成功转换: {converted_count} 个文件")
    print(f"跳过文件: {skipped_count} 个文件")
    print(f"失败文件: {error_count} 个文件")


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("图片格式转换工具 - 转换为 JPG")
    print("=" * 50)
    print()
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <图片文件夹路径> [--delete-original]")
        print()
        print("参数说明:")
        print("  <图片文件夹路径>  要处理的图片文件夹路径")
        print("  --delete-original  转换成功后删除原始文件（可选）")
        print()
        print("示例:")
        print(f"  python {sys.argv[0]} /path/to/images")
        print(f"  python {sys.argv[0]} ./my_images --delete-original")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    delete_original = '--delete-original' in sys.argv
    
    if delete_original:
        print("警告: 将在转换成功后删除原始文件!")
        response = input("确认继续? (yes/no): ")
        if response.lower() != 'yes':
            print("操作已取消")
            sys.exit(0)
    
    convert_images_to_jpg(folder_path, delete_original)