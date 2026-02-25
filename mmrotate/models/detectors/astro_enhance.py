import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LightweightSpatialAttention(nn.Module):
    """
    轻量级空间注意力模块：使用更小的卷积核和更简单的操作
    """
    def __init__(self, kernel_size=3):
        super(LightweightSpatialAttention, self).__init__()
        # 使用更小的卷积核减少参数
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 简化的空间注意力计算
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(attention)
        attention_mask = self.sigmoid(attention)
        
        return attention_mask 

class ImprovedAstroEnhancement(nn.Module):
    """
    改进的天文图像增强模块：专为天文图像中暗星增强设计的轻量级模块
    修复了可能导致NaN的操作
    """
    def __init__(self, in_channels=3):
        super(ImprovedAstroEnhancement, self).__init__()
        self.in_channels = in_channels
            
        # 暗星增强分支 - 使用深度可分离卷积减少参数量
        # 深度卷积
        self.depth_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 轻量级空间注意力
        self.spatial_attention = LightweightSpatialAttention()
        
        # 特征融合 - 使用1x1卷积减少参数量
        self.fusion_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        
        # 添加局部对比度增强
        self.local_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        
        # 原始分支
        original = x
        
        # 1. 使用安全的非线性变换增强暗区域，同时保留符号
        # 计算图像的平均亮度(绝对值平均)，以决定增强程度
        abs_mean = torch.mean(torch.abs(x), dim=[2, 3], keepdim=True) + 1e-5
        
        # 使用符号保留函数: y = sign(x) * f(|x|)
        x_sign = torch.sign(x)
        x_abs = torch.abs(x)
        
        # 根据平均亮度动态调整增强强度
        enhance_scale = torch.clamp(1.0 / (abs_mean + 0.1), 0.5, 3.0)
        
        # 对暗区域用非线性函数增强，保留原始符号
        # 使用 f(x) = x * (1 - e^(-ax)) 形式的函数，针对小值有更强的增强效果
        enhanced = x_sign * x_abs * (1.0 - torch.exp(-3.0 * enhance_scale * x_abs))
        
        # 2. 局部对比度增强 (修改为更安全的方式)
        local_mean = self.local_pool(enhanced) + 1e-5  # 防止除零
        # 使用tanh限制增强因子的范围
        enhance_factor = torch.tanh(-local_mean * 2)
        enhanced = enhanced * (1 + enhance_factor)
        
        # 3. 轻量级深度可分离卷积提取暗星特征
        enhanced = self.depth_conv(enhanced)
        enhanced = F.leaky_relu(enhanced, 0.1)
        enhanced = self.point_conv(enhanced)
        
        # 4. 应用轻量级空间注意力，聚焦在暗星上
        attention_mask = self.spatial_attention(enhanced)
        enhanced = enhanced * attention_mask
        
        # 5. 特征融合 (concat方式)
        concat_feat = torch.cat([original, enhanced], dim=1)
        output = self.fusion_conv(concat_feat)
        
        return output