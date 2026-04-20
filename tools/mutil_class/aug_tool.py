import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2

# 设置输入和输出文件夹
input_folder = "path_to_fits_folder"  # 替换为你的 FITS 文件夹路径
output_folder = "path_to_output_folder"  # 替换为输出 PNG 图像的文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def process_fits_to_png(fits_path, output_path):
    # 读取 FITS 文件
    with fits.open(fits_path) as hdul:
        data = hdul[0].data  # 读取主数据

    # 确保数据是 2D 图像
    if len(data.shape) == 3:  # 如果是 3 通道，选取合适的通道
        data = np.mean(data, axis=0)  # 计算均值通道

    # 取对数增强，防止对数变换出现负值
    data = np.log1p(np.maximum(data, 0))  

    # 归一化到 0-255
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)  # 转换为 8-bit 图像

    # 保存为 PNG
    cv2.imwrite(output_path, data)

# 遍历文件夹中的 FITS 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".fits") or filename.endswith(".fit"):
        fits_path = os.path.join(input_folder, filename)
        png_path = os.path.join(output_folder, filename.replace(".fits", ".png").replace(".fit", ".png"))
        process_fits_to_png(fits_path, png_path)
        print(f"Processed: {fits_path} -> {png_path}")

print("All FITS files have been processed and saved as PNG.")
