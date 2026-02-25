import torch
import torch.nn as nn
import torch.nn.functional as F


class AstroImageEnhancement(nn.Module):
    """
    用于天文图像的增强模块，专门针对暗星检测优化
    实现了多种传统增强方法的PyTorch版本
    """
    def __init__(self, 
                 gamma=0.3, 
                 log_c=1.0, 
                 sharpen_strength=0.5,
                 contrast_alpha=1.2,
                 brightness_beta=0.02):
        super().__init__()
        self.gamma = gamma
        self.log_c = log_c
        self.sharpen_strength = sharpen_strength
        self.contrast_alpha = contrast_alpha
        self.brightness_beta = brightness_beta
        
        # 定义锐化卷积核 - 会自动移动到与模型相同的设备
        sharpen_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer('sharpen_kernel', sharpen_kernel)
        
        # 定义用于形态学操作的卷积核（圆形结构元素近似）
        morph_kernel = torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer('morph_kernel', morph_kernel)

    def white_balance(self, img):
        """灰度世界白平衡"""
        gray_mean = torch.mean(img, dim=(2, 3), keepdim=True)
        scale = gray_mean / (torch.mean(gray_mean, dim=1, keepdim=True) + 1e-6)
        return torch.clamp(img / scale, 0, 1)
    
    def adaptive_hist_eq(self, img):
        """自适应直方图均衡化的简化近似"""
        # 首先转为灰度图
        if img.shape[1] == 3:
            # RGB转灰度
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
            
        # 局部归一化来模拟CLAHE效果
        # 使用平均池化获取局部均值
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        # 使用局部方差
        local_var = F.avg_pool2d(gray ** 2, kernel_size=9, stride=1, padding=4) - local_mean ** 2
        local_std = torch.sqrt(torch.clamp(local_var, min=1e-6))
        
        # 归一化
        normalized = (gray - local_mean) / (local_std + 1e-6)
        enhanced_gray = torch.clamp(normalized * 0.2 + 0.5, 0, 1)
        
        # 如果输入是彩色图像，保持色彩但调整亮度
        if img.shape[1] == 3:
            hsv_ratio = enhanced_gray / (gray + 1e-6)
            enhanced = img * hsv_ratio
            return torch.clamp(enhanced, 0, 1)
        else:
            return enhanced_gray
    
    def gaussian_blur(self, img, kernel_size=3, sigma=0.5):
        """高斯模糊"""
        # 创建高斯核
        channels = img.shape[1]
        device = img.device
        
        # 使用分离的1D卷积实现高斯模糊
        # 水平方向
        kernel_h = self.create_gaussian_kernel(kernel_size, sigma).repeat(channels, 1, 1, 1).to(device)
        blurred_h = F.conv2d(
            F.pad(img, (kernel_size//2, kernel_size//2, 0, 0), mode='reflect'),
            kernel_h,
            groups=channels
        )
        
        # 垂直方向
        kernel_v = self.create_gaussian_kernel(kernel_size, sigma).transpose(2, 3).repeat(channels, 1, 1, 1).to(device)
        blurred = F.conv2d(
            F.pad(blurred_h, (0, 0, kernel_size//2, kernel_size//2), mode='reflect'),
            kernel_v,
            groups=channels
        )
        
        return blurred
    
    def create_gaussian_kernel(self, kernel_size, sigma):
        """创建高斯核"""
        # 创建在CPU上，稍后会在使用时移动到正确设备
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # 转为2D卷积核
        kernel_2d = kernel_1d.view(1, 1, kernel_size, 1)
        return kernel_2d
    
    def log_transform(self, img):
        """对数变换增强暗区"""
        return self.log_c * torch.log1p(img) / torch.log(torch.tensor(2.0))
    
    def gamma_correction(self, img):
        """伽马校正"""
        return torch.pow(img, self.gamma)
    
    def contrast_brightness(self, img):
        """调整对比度和亮度"""
        return torch.clamp(img * self.contrast_alpha + self.brightness_beta, 0, 1)
    
    def sharpen(self, img):
        """锐化处理"""
        # 用分组卷积实现每个通道独立锐化
        img_pad = F.pad(img, (1, 1, 1, 1), mode='reflect')
        # 确保卷积核与输入在同一设备上
        sharpened = F.conv2d(img_pad, self.sharpen_kernel.to(img.device), groups=3)
        
        # 混合原图和锐化结果
        return torch.clamp(img + self.sharpen_strength * (sharpened - img), 0, 1)
    
    def star_enhance(self, img):
        """星点增强 - 使用形态学操作"""
        # 进行灰度转换
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
            
        # 使用形态学顶帽操作增强星点
        # 先进行膨胀
        dilated = F.max_pool2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), kernel_size=3, stride=1)
        # 顶帽 = 原图 - 开运算(原图) ≈ 原图 - 膨胀(原图)
        tophat = torch.clamp(gray - dilated, 0, 1)
        
        # 将顶帽结果加回原图
        if img.shape[1] == 3:
            enhanced = img + tophat.repeat(1, 3, 1, 1) * 0.3
        else:
            enhanced = img + tophat * 0.3
            
        return torch.clamp(enhanced, 0, 1)
    
    def normalize_minmax(self, img):
        """最小-最大归一化"""
        b, c, h, w = img.shape
        min_vals = img.view(b, c, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        max_vals = img.view(b, c, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        return (img - min_vals) / (max_vals - min_vals + 1e-6)

    def forward(self, x):
        """
        增强流程：去噪 → 白平衡 → 对数变换 → 伽马校正 → 星点增强 → 对比度调整 → 锐化
        """
        with torch.no_grad():  # 传统方法不参与梯度计算
            # 高斯模糊去噪
            blurred = self.gaussian_blur(x)
            
            # 白平衡
            wb = self.white_balance(blurred)
            
            # 自适应直方图均衡化
            clahe = self.adaptive_hist_eq(wb)
            
            # 对数变换增强暗区
            log_enhanced = self.log_transform(clahe)
            log_enhanced = self.normalize_minmax(log_enhanced)
            
            # 伽马校正进一步增强暗区
            gamma_corrected = self.gamma_correction(log_enhanced)
            
            # 星点增强
            star_enhanced = self.star_enhance(gamma_corrected)
            
            # 对比度和亮度调整
            contrast_adjusted = self.contrast_brightness(star_enhanced)
            
            # 锐化增强星点边缘
            sharpened = self.sharpen(contrast_adjusted)
            
        return sharpened  # 最终输出增强后的图像 [B,3,H,W]


class AstroEnhancementPipeline(nn.Module):
    """可配置的天文图像增强流水线"""
    def __init__(self, use_steps=None, **kwargs):
        """
        参数:
            use_steps: 要使用的处理步骤列表，None表示使用全部
                可选值: ['denoise', 'white_balance', 'clahe', 'log', 'gamma', 
                         'star_enhance', 'contrast', 'sharpen']
            **kwargs: 传递给AstroImageEnhancement的参数
        """
        super().__init__()
        self.enhancer = AstroImageEnhancement(**kwargs)
        
        # 默认使用所有步骤
        if use_steps is None:
            use_steps = ['denoise', 'white_balance', 'clahe', 'log', 'gamma', 
                         'star_enhance', 'contrast', 'sharpen']
        self.use_steps = use_steps
        
    def forward(self, x):
        """根据配置的步骤顺序执行增强"""
        with torch.no_grad():
            img = x
            
            # 应用选定的增强步骤
            if 'denoise' in self.use_steps:
                img = self.enhancer.gaussian_blur(img)
                
            if 'white_balance' in self.use_steps:
                img = self.enhancer.white_balance(img)
                
            if 'clahe' in self.use_steps:
                img = self.enhancer.adaptive_hist_eq(img)
                
            if 'log' in self.use_steps:
                img = self.enhancer.log_transform(img)
                img = self.enhancer.normalize_minmax(img)
                
            if 'gamma' in self.use_steps:
                img = self.enhancer.gamma_correction(img)
                
            if 'star_enhance' in self.use_steps:
                img = self.enhancer.star_enhance(img)
                
            if 'contrast' in self.use_steps:
                img = self.enhancer.contrast_brightness(img)
                
            if 'sharpen' in self.use_steps:
                img = self.enhancer.sharpen(img)
                
        return img

