import os
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# 导入你的模型
from astro_enhancement_model import ImprovedAstroEnhancement, ImprovedStarEnhancementModule

def process_directory(input_dir, output_dir, model_path, device='cuda', img_size=None):
    """
    批量处理指定目录下的所有图像，并保存增强结果及可视化对比图
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出结果目录
        model_path: 训练好的模型权重路径
        device: 使用的设备 ('cuda' 或 'cpu')
        img_size: 可选，处理图像的尺寸，如果为None则保持原始尺寸
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'originals'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'enhanced'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'star_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'final_outputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    # 加载模型
    model = ImprovedAstroEnhancement(in_channels=3)
    
    # 如果模型权重路径存在，加载权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"成功加载模型: {model_path}")
    else:
        print(f"警告: 找不到模型权重 {model_path}，使用未训练的模型")
    
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 获取图像文件列表
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.fits'))]
    
    if not image_files:
        print(f"错误: 在 {input_dir} 中没有找到有效的图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件，开始处理...")
    
    # 批处理所有图像
    for img_file in tqdm(image_files):
        # 拼接完整路径
        img_path = os.path.join(input_dir, img_file)
        
        try:
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            
            # 如果指定了图像尺寸，则调整大小
            if img_size is not None:
                transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor()
                ])
            else:
                transform = transforms.ToTensor()
            
            # 转换为张量
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # 使用模型处理图像
            with torch.no_grad():
                # 访问内部模块以获取中间结果
                enhanced, star_map = model.enhancement_module(img_tensor)
                final_output = model(img_tensor)
            
            # 去掉批次维度
            img_tensor = img_tensor.squeeze(0)
            enhanced = enhanced.squeeze(0)
            star_map = star_map.squeeze(0)
            final_output = final_output.squeeze(0)
            # print("img_tensor", img_tensor)
            # print('enhanced',enhanced)
            # print('final_output', final_output)
            
            # 保存原始图像
            filename_no_ext = os.path.splitext(img_file)[0]
            save_image(img_tensor, os.path.join(output_dir, 'originals', f"{filename_no_ext}_original.png"))
            
            # 保存增强后图像
            save_image(enhanced, os.path.join(output_dir, 'enhanced', f"{filename_no_ext}_enhanced.png"))
            
            # 保存星体检测图
            save_image(star_map, os.path.join(output_dir, 'star_maps', f"{filename_no_ext}_star_map.png"))
            
            # 保存最终输出
            save_image(final_output, os.path.join(output_dir, 'final_outputs', f"{filename_no_ext}_final.png"))
            
            # 创建对比可视化
            create_comparison_visualization(
                img_tensor.cpu(), 
                enhanced.cpu(), 
                star_map.cpu(), 
                final_output.cpu(),
                os.path.join(output_dir, 'comparisons', f"{filename_no_ext}_comparison.png")
            )
            
        except Exception as e:
            print(f"处理 {img_file} 时出错: {str(e)}")
    
    print(f"处理完成! 结果保存在 {output_dir}")

def create_comparison_visualization(original, enhanced, star_map, final, save_path):
    """
    创建包含原始图像、星体检测图、增强图像和最终输出的对比可视化
    
    Args:
        original: 原始图像张量 (C, H, W)
        enhanced: 增强图像张量 (C, H, W)
        star_map: 星体检测图张量 (1, H, W)
        final: 最终输出张量 (C, H, W)
        save_path: 保存路径
    """
    # 将张量转换为NumPy数组
    original_np = original.permute(1, 2, 0).numpy()
    enhanced_np = enhanced.permute(1, 2, 0).numpy()
    star_map_np = star_map.permute(1, 2, 0).numpy()
    final_np = final.permute(1, 2, 0).numpy()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 显示原始图像
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('origin', fontsize=14)
    axes[0, 0].axis('off')
    
    # 显示星体检测图
    axes[0, 1].imshow(star_map_np.squeeze(), cmap='plasma')
    axes[0, 1].set_title('star_detection', fontsize=14)
    axes[0, 1].axis('off')
    
    # 显示增强图像
    axes[1, 0].imshow(enhanced_np, cmap='plasma')
    axes[1, 0].set_title('enhance', fontsize=14)
    axes[1, 0].axis('off')
    
    # 显示最终输出
    axes[1, 1].imshow(final_np, cmap='gray')
    axes[1, 1].set_title('final', fontsize=14)
    axes[1, 1].axis('off')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def create_batch_comparison(output_dir, max_images=5):
    """
    创建包含多个图像处理结果的批量对比可视化
    
    Args:
        output_dir: 输出目录
        max_images: 最多显示的图像数量
    """
    originals_dir = os.path.join(output_dir, 'originals')
    original_files = sorted([f for f in os.listdir(originals_dir) if f.endswith('.png')])
    
    if not original_files:
        print("没有找到处理后的图像文件")
        return
    
    # 限制图像数量
    original_files = original_files[:max_images]
    n_images = len(original_files)
    
    # 创建图形
    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4*n_images))
    
    # 如果只有一张图像，确保axes是二维的
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    # 设置列标题
    column_titles = ['原始图像', '星体检测图', '增强图像', '最终输出']
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontsize=14)
    
    # 填充每一行
    for i, img_file in enumerate(original_files):
        basename = os.path.splitext(img_file)[0].replace('_original', '')
        
        # 加载图像
        original = plt.imread(os.path.join(output_dir, 'originals', img_file))
        star_map = plt.imread(os.path.join(output_dir, 'star_maps', basename + '_star_map.png'))
        enhanced = plt.imread(os.path.join(output_dir, 'enhanced', basename + '_enhanced.png'))
        final = plt.imread(os.path.join(output_dir, 'final_outputs', basename + '_final.png'))
        
        # 显示图像
        axes[i, 0].imshow(original)
        axes[i, 0].axis('off')
        
        if len(star_map.shape) == 3 and star_map.shape[2] > 1:
            axes[i, 1].imshow(star_map)
        else:
            axes[i, 1].imshow(star_map, cmap='plasma')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(enhanced)
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(final)
        axes[i, 3].axis('off')
        
        # 添加图像名称
        axes[i, 0].set_ylabel(basename, fontsize=12, rotation=0, labelpad=50, ha='right')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"批量对比图已保存到 {os.path.join(output_dir, 'batch_comparison.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='天文图像批量增强处理')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出结果目录')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='使用的设备 (cuda/cpu)')
    parser.add_argument('--img_size', type=int, default=None, help='处理图像的尺寸，默认保持原始尺寸')
    parser.add_argument('--batch_vis', action='store_true', help='是否创建批量对比可视化')
    
    args = parser.parse_args()
    
    # 处理图像
    process_directory(args.input_dir, args.output_dir, args.model_path, args.device, args.img_size)
    
    # 如果需要，创建批量对比可视化
    if args.batch_vis:
        create_batch_comparison(args.output_dir)