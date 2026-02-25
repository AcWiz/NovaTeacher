import torch
import torch.nn as nn  # 导入 nn 模块
import torch.nn.functional as F  # 导入 F 模块

# class WaterNet(nn.Module):
#     """水下图像增强网络
#     输入: [B, 3, H, W] (原始图像)
#     输出: [B, 3, H, W] (增强后的图像)
#     """
#     def __init__(self):
#         super().__init__()
#         self.traditional = TraditionalEnhancements()
        
#         # 编码器（下采样）
#         self.encoder = nn.Sequential(
#             nn.Conv2d(12, 64, 3, padding=1),  # 输入12通道
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)  # 分辨率减半 [H/2, W/2]
#         )
        
#         # 残差块（保持分辨率）
#         self.res_blocks = nn.Sequential(
#             ResBlock(64), ResBlock(64), ResBlock(64)
#         )
        
#         # 解码器（上采样恢复分辨率）
#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 分辨率恢复 [H, W]
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 3, 3, padding=1),
#             nn.Sigmoid()  # 输出归一化到[0,1]
#         )

#     def forward(self, x):
#         x_orig = x  # [B,3,H,W]
#         features = self.traditional(x)    # [B,12,H,W]
#         x = self.encoder(features)        # [B,64,H/2,W/2]
#         x = self.res_blocks(x)            # [B,64,H/2,W/2]
#         x = self.decoder(x)               # [B,3,H,W] (恢复分辨率)
#         return torch.clamp(x + x_orig, 0, 1)  # 残差连接

# class ResBlock(nn.Module):
#     """残差块（输入输出通道数相同）
#     输入: [B, C, H, W]
#     输出: [B, C, H, W]
#     """
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels)
#         )
        
#     def forward(self, x):
#         return x + self.conv(x)  # 残差连接
    


# class TraditionalEnhancements(nn.Module):
#     """传统图像增强模块，生成多通道输入（用于Water-Net）
#     输入: [B, 3, H, W] (原始图像)
#     输出: [B, 12, H, W] (原始 + 白平衡 + 直方图均衡 + 伽马校正)
#     """
#     def __init__(self):
#         super().__init__()
        
#     def white_balance(self, img):
#         """灰度世界白平衡"""
#         # 输入: [B, 3, H, W]
#         # 输出: [B, 3, H, W]
#         gray_mean = torch.mean(img, dim=(2, 3), keepdim=True)
#         scale = gray_mean / (torch.mean(gray_mean, dim=1, keepdim=True) + 1e-6)
#         return torch.clamp(img / scale, 0, 1)

#     def gamma_correction(self, img, gamma=0.5):
#         """伽马校正"""
#         # 输入: [B, 3, H, W]
#         # 输出: [B, 3, H, W]
#         return torch.pow(img, gamma)

#     def hist_equalization(self, img):
#         """直方图均衡化近似"""
#         # 输入: [B, 3, H, W]
#         # 输出: [B, 3, H, W]
#         mean = torch.mean(img, dim=(2, 3), keepdim=True)
#         std = torch.std(img, dim=(2, 3), keepdim=True)
#         return torch.clamp((img - mean) / (std + 1e-6) * 0.2 + 0.5, 0, 1)

#     def forward(self, x):
#         # 输入: [B, 3, H, W]
#         # 输出: [B, 12, H, W] (原始 + WB + HE + GC)
#         with torch.no_grad():  # 传统方法不参与梯度计算
#             wb = self.white_balance(x)     # [B,3,H,W]
#             gc = self.gamma_correction(x)  # [B,3,H,W]
#             he = self.hist_equalization(x) # [B,3,H,W]
#         return torch.cat([x, wb, he, gc], dim=1)  # 通道拼接


class TraditionalEnhancements(nn.Module):
    """顺序执行三种传统增强方法"""
    def __init__(self, gamma=0.5):
        super().__init__()
        self.gamma = gamma
    
    def white_balance(self, img):
        """灰度世界白平衡"""
        gray_mean = torch.mean(img, dim=(2,3), keepdim=True)
        scale = gray_mean / (torch.mean(gray_mean, dim=1, keepdim=True) + 1e-6)
        return torch.clamp(img / scale, 0, 1)

    def gamma_correction(self, img):
        """伽马校正"""
        return torch.pow(img, self.gamma)

    def hist_equalization(self, img):
        """直方图均衡化近似"""
        mean = torch.mean(img, dim=(2,3), keepdim=True)
        std = torch.std(img, dim=(2,3), keepdim=True)
        return torch.clamp((img - mean)/(std + 1e-6)*0.2 + 0.5, 0, 1)

    def forward(self, x):
        """增强流程：白平衡 → 伽马校正 → 直方图均衡"""
        with torch.no_grad():  # 传统方法不参与梯度计算
            # 按顺序执行增强
            wb = self.white_balance(x)          # 白平衡
            gc = self.gamma_correction(wb)      # 在wb基础上做伽马校正
            he = self.hist_equalization(gc)     # 在gc基础上做直方图均衡
        return he  # 最终输出增强后的图像 [B,3,H,W]




class CSAM(nn.Module):
    """修改后的CSAM模块（输出3通道）"""
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        # 通道注意力部分（保持原设计）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//4),
            nn.ReLU(),
            nn.Linear(in_channels//4, in_channels),
            nn.Sigmoid()
        )
        
        # 新增通道压缩层（6→3）
        self.channel_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # 通道注意力加权
        att = self.avg_pool(x).view(b, c)       # [B, C]
        att = self.fc(att).view(b, c, 1, 1)     # [B, C, 1, 1]
        weighted = x * att.expand_as(x)          # [B, C, H, W]
        
        # 通道压缩
        output = self.channel_reduce(weighted)   # [B, 3, H, W]
        return output



# class CSAM(nn.Module):
#     """修改后的CSAM模块（输出3通道）"""
#     def __init__(self, in_channels=6, out_channels=3):
#         super().__init__()
        
#         # 通道注意力部分（保持原设计）
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels//4),
#             nn.ReLU(),
#             nn.Linear(in_channels//4, in_channels),
#             nn.Sigmoid()
#         )
        
#         # 新增通道压缩层（6→3）
#         self.channel_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
#     def forward(self, x1, x2=None):
#         b, c, h, w = x1.size()
        
#         # 如果有增强图像，将其和原图合并
#         if x2 is not None:
#             x = torch.cat([x1, x2], dim=1)  # 合并原图和增强图像，维度变为 [B, 2C, H, W]
#         else:
#             x = x1
        
#         # 通道注意力加权
#         att = self.avg_pool(x).view(b, c * 2)  # 修改这里，适应输入合并后的通道数
#         att = self.fc(att).view(b, c * 2, 1, 1)  # [B, 2C, 1, 1]
#         weighted = x * att.expand_as(x)  # [B, 2C, H, W]
        
#         # 通道压缩
#         output = self.channel_reduce(weighted)  # [B, 3, H, W]
#         return output





class SharpenLayer(nn.Module):
    """锐化卷积层
    输入: [B, 3, H, W] (原始图像)
    输出: [B, 3, H, W] (锐化后的图像)
    """
    def __init__(self, sharpen_lambda=9):
        super().__init__()
        # 定义锐化核（固定参数）
        self.kernel = nn.Parameter(
            torch.tensor([
                [-1, -1, -1],
                [-1, sharpen_lambda, -1],
                [-1, -1, -1]
            ], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False  # 核参数不参与训练
        )

    def forward(self, x):
        # 输入: [B, 3, H, W]
        # 输出: [B, 3, H, W]
        channels = []
        for c in range(x.shape[1]):
            # 对每个通道单独卷积
            channel = F.conv2d(x[:, c:c+1], self.kernel, padding=1)  # 保持分辨率
            channels.append(channel)
        return torch.clamp(torch.cat(channels, dim=1), 0, 1)  # 合并通道并截断