import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# 设置文件夹路径
png_folder = "/home/flh/datasets/LAMOST_new/dataset_ori/train/images"       # 存放 PNG 图片的文件夹（用于匹配文件名）
fits_folder = "/home/flh/datasets/LAMOST_new/images_1k"      # 存放 FITS 文件的文件夹
output_folder = "/home/flh/datasets/LAMOST_new/dataset_ori/train/images_lg2"  # 输出 PNG 的文件夹




# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def process_fits_to_grayscale_png(fits_path, output_path):
    """ 读取 FITS 文件并转换为灰度 PNG（使用 matplotlib） """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data  # 读取 FITS 数据

    # 确保数据是 2D（如果是 3D，取均值）
    if len(data.shape) == 3:
        data = np.mean(data, axis=0)  # 计算通道均值，转换成 2D 灰度图

    # 取对数增强，并防止负值
    data = np.log1p(np.maximum(data, 0))

    # 归一化到 0-1（matplotlib 可以自动缩放到 0-255）
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # 使用 matplotlib 保存为灰度 PNG
    plt.imsave(output_path, data, cmap='gray', format='png')

# 遍历 PNG 文件夹中的文件
for filename in os.listdir(png_folder):
    if filename.endswith(".png"):
        base_name = os.path.splitext(filename)[0]  # 获取不带扩展名的文件名
        fits_path = os.path.join(fits_folder, base_name + ".fit")

        if os.path.exists(fits_path):  # 检查 FITS 文件是否存在
            output_path = os.path.join(output_folder, base_name + ".png")
            process_fits_to_grayscale_png(fits_path, output_path)
            print(f"Processed: {fits_path} -> {output_path}")
        else:
            print(f"Warning: FITS file not found for {filename}")

print("All matching FITS files have been processed and saved as grayscale PNG.")
