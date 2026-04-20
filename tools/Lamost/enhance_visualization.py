def visualize_and_save(original, enhanced, fused, output_path, filename):
    """
    可视化增强效果并保存
    
    Args:
        original: 原始图像张量
        enhanced: 增强后图像张量
        fused: 融合后图像张量
        output_path: 输出路径
        filename: 文件名（不含扩展名）
    """
    # 确保输出路径存在
    os.makedirs(output_path, exist_ok=True)
    
    # 移动到CPU并移除批次维度
    original = original.cpu().squeeze()
    enhanced = enhanced.cpu().squeeze()
    fused = fused.cpu().squeeze()
    
    # 处理通道数
    if len(original.shape) == 2:  # 如果是单通道
        original = original.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)
        fused = fused.unsqueeze(0)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原始图像
    if original.shape[0] == 1:
        axes[0].imshow(original[0], cmap='gray')
    else:
        axes[0].imshow(original.permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示增强后图像
    if enhanced.shape[0] == 1:
        axes[1].imshow(enhanced[0], cmap='gray')
    else:
        axes[1].imshow(enhanced.permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title('增强后图像')
    axes[1].axis('off')
    
    # 显示融合后图像
    if fused.shape[0] == 1:
        axes[2].imshow(fused[0], cmap='gray')
    else:
        axes[2].imshow(fused.permute(1, 2, 0).clamp(0, 1))
    axes[2].set_title('融合后图像')
    axes[2].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(os.path.join(output_path, f"{filename}_comparison.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    # 单独保存各图像为图像文件
    if original.shape[0] == 3:  # RGB图像
        save_image(original, os.path.join(output_path, f"{filename}_original.png"))
        save_image(enhanced, os.path.join(output_path, f"{filename}_enhanced.png"))
        save_image(fused, os.path.join(output_path, f"{filename}_fused.png"))
    else:  # 灰度图像
        save_image(original, os.path.join(output_path, f"{filename}_original.png"))
        save_image(enhanced, os.path.join(output_path, f"{filename}_enhanced.png"))
        save_image(fused, os.path.join(output_path, f"{filename}_fused.png"))
        
import os
import glob
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# 导入你的模型类（确保该文件能够被导入）
from astro_enhancement_model import ImprovedAstroEnhancement

def load_model(model_path, in_channels=3):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重文件路径
        in_channels: 输入通道数
        
    Returns:
        加载好权重的模型和设备
    """
    # 创建模型实例
    model = ImprovedAstroEnhancement(in_channels=in_channels)
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载预训练权重
    try:
        if device.type == 'cuda':
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 使用 strict=False 以允许模型架构的轻微变化
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"模型权重加载失败: {e}")
        raise
    
    # 设置为评估模式
    model.eval()
    
    # 移动到设备
    model = model.to(device)
    
    return model, device

def process_image(model, image_path, device, in_channels=3):
    """
    处理单张图像并获取增强结果
    
    Args:
        model: 训练好的模型
        image_path: 图像路径
        device: 计算设备
        in_channels: 输入通道数
        
    Returns:
        原始图像、增强后图像、星体图、最终输出的张量
    """
    # 加载图像
    img = Image.open(image_path).convert('RGB')  # 明确转换为RGB格式
    
    # 转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 应用变换
    img_tensor = transform(img).unsqueeze(0)  # 添加batch维度
    
    # 确保通道数匹配
    if img_tensor.shape[1] == 1 and in_channels == 3:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
    elif img_tensor.shape[1] == 3 and in_channels == 1:
        # 转换为灰度
        img_tensor = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    elif img_tensor.shape[1] == 4:  # 处理RGBA图像
        img_tensor = img_tensor[:, :3]  # 只保留RGB通道
    
    # 移动到设备
    img_tensor = img_tensor.to(device)
    
    # 获取增强模块和输出
    with torch.no_grad():
        # 获取增强模块
        enhancement_module = model.enhancement_module
        
        # 应用增强模块获取中间结果
        enhanced, star_map = enhancement_module(img_tensor)
        
        # 获取最终融合输出
        final_output = model(img_tensor)
    
    return img_tensor, enhanced, star_map, final_output

def visualize_and_save(original, enhanced, star_map, final_output, output_path, filename):
    """
    可视化增强效果并保存
    
    Args:
        original: 原始图像张量
        enhanced: 增强后图像张量
        star_map: 星体检测图张量
        final_output: 最终输出张量
        output_path: 输出路径
        filename: 文件名（不含扩展名）
    """
    # 确保输出路径存在
    os.makedirs(output_path, exist_ok=True)
    
    # 移动到CPU并移除批次维度
    original = original.cpu().squeeze()
    enhanced = enhanced.cpu().squeeze()
    star_map = star_map.cpu().squeeze()
    final_output = final_output.cpu().squeeze()
    
    # 处理通道数
    if len(original.shape) == 2:  # 如果是单通道
        original = original.unsqueeze(0)
        enhanced = enhanced.unsqueeze(0)
        final_output = final_output.unsqueeze(0)
    
    # 创建图形
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 显示原始图像
    if original.shape[0] == 1:
        axes[0].imshow(original[0], cmap='gray')
    else:
        axes[0].imshow(original.permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示增强后图像
    if enhanced.shape[0] == 1:
        axes[1].imshow(enhanced[0], cmap='gray')
    else:
        axes[1].imshow(enhanced.permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title('增强后图像')
    axes[1].axis('off')
    
    # 显示星体检测图
    axes[2].imshow(star_map, cmap='inferno')
    axes[2].set_title('星体检测图')
    axes[2].axis('off')
    
    # 计算原图和增强图的融合
    fusion_weight = 0.5  # 可以调整融合权重
    if original.shape[0] == 3:  # RGB图像
        fused_image = original * (1 - fusion_weight) + enhanced * fusion_weight
    else:  # 灰度图像
        fused_image = original * (1 - fusion_weight) + enhanced * fusion_weight
    
    # 显示融合后的图像
    if fused_image.shape[0] == 1:
        axes[3].imshow(fused_image[0], cmap='gray')
    else:
        axes[3].imshow(fused_image.permute(1, 2, 0).clamp(0, 1))
    axes[3].set_title('原图与增强图融合')
    axes[3].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(os.path.join(output_path, f"{filename}_comparison.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    # 单独保存各图像为图像文件
    if original.shape[0] == 3:  # RGB图像
        save_image(original, os.path.join(output_path, f"{filename}_original.png"))
        save_image(enhanced, os.path.join(output_path, f"{filename}_enhanced.png"))
        save_image(star_map.unsqueeze(0), os.path.join(output_path, f"{filename}_star_map.png"))
        save_image(fused_image, os.path.join(output_path, f"{filename}_fused.png"))
    else:  # 灰度图像
        save_image(original, os.path.join(output_path, f"{filename}_original.png"))
        save_image(enhanced, os.path.join(output_path, f"{filename}_enhanced.png"))
        save_image(star_map.unsqueeze(0), os.path.join(output_path, f"{filename}_star_map.png"))
        save_image(fused_image, os.path.join(output_path, f"{filename}_fused.png"))

def process_directory(model, input_dir, output_dir, device, in_channels=3, debug=False):
    """
    批量处理目录中的所有图像
    
    Args:
        model: 训练好的模型
        input_dir: 输入图像目录
        output_dir: 输出结果目录
        device: 计算设备
        in_channels: 输入通道数
        debug: 是否启用调试模式
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f"发现 {len(image_files)} 个图像文件...")
    
    # 处理每个图像
    success_count = 0
    failed_images = []
    
    for i, image_path in enumerate(image_files):
        try:
            # 获取文件名（不含路径和扩展名）
            filename = os.path.splitext(os.path.basename(image_path))[0]
            print(f"处理图像 {i+1}/{len(image_files)}: {filename}")
            
            if debug:
                # 调试模式：显示图像信息
                img = Image.open(image_path)
                print(f"  原始图像格式: {img.format}, 尺寸: {img.size}, 模式: {img.mode}")
                img_rgb = img.convert('RGB')
                print(f"  转RGB后: 尺寸: {img_rgb.size}, 模式: {img_rgb.mode}")
            
            # 处理图像
            original, enhanced, star_map, final_output = process_image(
                model, image_path, device, in_channels)
            
            if debug:
                print(f"  图像张量形状: {original.shape}")
            
            # 可视化并保存结果
            visualize_and_save(
                original, enhanced, star_map, final_output, output_dir, filename)
            
            success_count += 1
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            failed_images.append((image_path, str(e)))
            continue
    
    # 打印处理汇总
    print(f"\n处理完成! 成功: {success_count}/{len(image_files)}, 失败: {len(failed_images)}")
    if failed_images:
        print("\n失败图像列表:")
        for img_path, error in failed_images:
            print(f"  - {img_path}: {error}")
    
    print(f"\n结果保存至: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='天文图像增强可视化')
    parser.add_argument('--model', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出结果目录')
    parser.add_argument('--channels', type=int, default=3, help='输入通道数 (1=灰度, 3=RGB)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--single_file', type=str, default=None, help='处理单个文件而不是整个目录')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU进行处理')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    try:
        model, device = load_model(args.model, args.channels)
        print(f"模型已加载到 {device}")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return
    
    # 处理图像
    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f"错误: 文件 '{args.single_file}' 不存在")
            return
            
        try:
            # 确保输出目录存在
            os.makedirs(args.output_dir, exist_ok=True)
            
            # 获取文件名（不含路径和扩展名）
            filename = os.path.splitext(os.path.basename(args.single_file))[0]
            print(f"处理单个图像: {filename}")
            
            # 处理图像
            original, enhanced, star_map, final_output = process_image(
                model, args.single_file, device, args.channels)
            
            # 可视化并保存结果
            visualize_and_save(
                original, enhanced, star_map, final_output, args.output_dir, filename)
            
            print(f"处理完成! 结果保存至 {args.output_dir}")
            
        except Exception as e:
            print(f"处理图像失败: {str(e)}")
    else:
        # 处理整个目录
        process_directory(model, args.input_dir, args.output_dir, device, args.channels, args.debug)

if __name__ == "__main__":
    main()