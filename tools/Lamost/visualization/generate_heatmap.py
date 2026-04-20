import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse

def generate_heatmap(image_path, output_path=None, high_brightness_ratio=0.2, low_brightness_ratio=0.05, threshold=127):
    """
    根据图像亮度生成热度图
    
    参数:
        image_path: 输入图像路径
        output_path: 输出热度图路径，默认为None（显示不保存）
        high_brightness_ratio: 高亮度区域选取点的比例
        low_brightness_ratio: 低亮度区域选取点的比例
        threshold: 区分高亮度和低亮度的阈值（0-255）
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 创建掩码区分高亮度和低亮度区域
    high_mask = gray >= threshold
    low_mask = gray < threshold
    
    # 获取高亮度和低亮度区域的坐标
    high_coords = np.column_stack(np.where(high_mask))
    low_coords = np.column_stack(np.where(low_mask))
    
    # 随机选择点
    np.random.seed(42)  # 设置随机种子以保证结果可重复
    
    # 计算需要选择的点数
    high_points_count = int(len(high_coords) * high_brightness_ratio)
    low_points_count = int(len(low_coords) * low_brightness_ratio)
    
    # 随机选择点
    selected_high_indices = np.random.choice(len(high_coords), high_points_count, replace=False) if len(high_coords) > 0 else []
    selected_low_indices = np.random.choice(len(low_coords), low_points_count, replace=False) if len(low_coords) > 0 else []
    
    selected_high_coords = high_coords[selected_high_indices] if len(high_coords) > 0 else np.array([])
    selected_low_coords = low_coords[selected_low_indices] if len(low_coords) > 0 else np.array([])
    
    # 合并所有选择的点
    all_selected_coords = np.vstack([selected_high_coords, selected_low_coords]) if len(selected_high_coords) > 0 and len(selected_low_coords) > 0 else (
        selected_high_coords if len(selected_high_coords) > 0 else selected_low_coords
    )
    
    # 创建热度图
    heatmap = np.zeros_like(gray, dtype=np.float32)
    
    # 为每个选择的点添加高斯核
    for y, x in all_selected_coords:
        # 高亮度区域的点权重更高
        weight = 1.0 if gray[y, x] >= threshold else 0.5
        # 添加高斯核
        sigma = 20  # 高斯核的标准差，控制热度扩散范围
        kernel_size = int(sigma * 3) * 2 + 1  # 核大小
        
        # 计算核的范围
        y_min, y_max = max(0, y - kernel_size//2), min(gray.shape[0], y + kernel_size//2 + 1)
        x_min, x_max = max(0, x - kernel_size//2), min(gray.shape[1], x + kernel_size//2 + 1)
        
        # 计算核内每个点的坐标
        for ky in range(y_min, y_max):
            for kx in range(x_min, x_max):
                # 计算到中心点的距离
                dist = np.sqrt((ky - y)**2 + (kx - x)**2)
                # 计算高斯值
                gaussian_value = weight * np.exp(-(dist**2) / (2 * sigma**2))
                print(gaussian_value)
                # 累加到热度图
                heatmap[ky, kx] += gaussian_value
    
    # 归一化热度图
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 显示原图
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示选择的点
    plt.subplot(1, 3, 2)
    point_map = np.zeros_like(gray)
    for y, x in all_selected_coords:
        point_map[y, x] = 255
    plt.imshow(point_map, cmap='gray')
    plt.title(f'选择的点 (高亮区域: {high_brightness_ratio}, 低亮区域: {low_brightness_ratio})')
    plt.axis('off')
    
    # 显示热度图
    plt.subplot(1, 3, 3)
    # 创建自定义颜色映射，从蓝色到红色
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    plt.imshow(heatmap, cmap=cmap)
    plt.colorbar(label='热度')
    plt.title('生成的热度图')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存或显示结果
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"热度图已保存至: {output_path}")
    else:
        plt.show()
    
    # 返回热度图数组
    return heatmap

def main():
    parser = argparse.ArgumentParser(description='根据图像亮度生成热度图')
    parser.add_argument('image_path', type=str, help='输入图像路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出热度图路径')
    parser.add_argument('--high-ratio', type=float, default=0.2, help='高亮度区域选取点的比例')
    parser.add_argument('--low-ratio', type=float, default=0.05, help='低亮度区域选取点的比例')
    parser.add_argument('--threshold', type=int, default=127, help='区分高亮度和低亮度的阈值（0-255）')
    
    args = parser.parse_args()
    
    generate_heatmap(
        args.image_path, 
        args.output, 
        args.high_ratio, 
        args.low_ratio, 
        args.threshold
    )

if __name__ == "__main__":
    main()