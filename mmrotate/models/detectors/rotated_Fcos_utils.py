
"""
增强代码版本二
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StarDetectionModule(nn.Module):
    """
    Specialized module for detecting stars in astronomical images.
    Uses a combination of Laplacian-of-Gaussian (LoG) inspired filters
    to detect point-like sources (stars) of various sizes.
    """
    def __init__(self, in_channels=3, feature_channels=16):
        super(StarDetectionModule, self).__init__()
        
        # Multi-scale star detection filters (inspired by LoG)
        # Use different kernel sizes to detect stars of different sizes
        self.star_filters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, feature_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True)
            ) for k in [3, 5, 7]  # Multiple scales for different star sizes
        ])
        
        # Combine star detection results
        self.combine = nn.Sequential(
            nn.Conv2d(feature_channels * 3, feature_channels, kernel_size=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # Star probability map generation
        self.star_map = nn.Sequential(
            nn.Conv2d(feature_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Apply star detection filters at multiple scales
        star_features = [filt(x) for filt in self.star_filters]
        
        # Combine features from different scales
        combined = torch.cat(star_features, dim=1)
        features = self.combine(combined)
        
        # Generate star probability map
        star_map = self.star_map(features)
        
        return star_map, features


class ImprovedStarEnhancementModule(nn.Module):
    """
    Improved star enhancement module that specifically targets dim stars
    while preserving the overall image structure.
    """
    def __init__(self, in_channels=3, feature_channels=24):
        super(ImprovedStarEnhancementModule, self).__init__()
        
        # Star detection module
        self.star_detector = StarDetectionModule(in_channels, feature_channels)
        
        # Background estimation network (use dilated convolutions to capture larger context)
        self.background_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=3, dilation=3),
            nn.Sigmoid()
        )
        
        # Enhancement parameters network
        self.enhancement_net = nn.Sequential(
            nn.Conv2d(feature_channels + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final output refinement
        self.output_refine = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Detect stars and get star features
        star_map, star_features = self.star_detector(x)
        
        # Estimate background (smoother version of the image)
        background = self.background_net(x)
        
        # Calculate enhancement factor based on star map and features
        # This determines how much to boost each pixel
        enhancement_params = self.enhancement_net(torch.cat([star_features, star_map], dim=1))
        
        # Apply a non-linear enhancement focused on dim stars
        # The key idea: enhance pixels more if they're slightly above background (dim stars)
        # but not too bright already (bright stars don't need enhancement)
        diff = torch.clamp(x - background, 0, 1)
        
        # This is the crucial part - we enhance dim stars (small diff values) more
        # The enhancement factor is higher for dim stars and lower for bright stars
        # We use an adaptive sigmoid-based approach: 
        # - star_map tells us where stars are likely to be
        # - enhancement_params controls the degree of enhancement
        # - diff ensures we focus on enhancing dim regions
        
        # Non-linear enhancement function targeting dim stars
        boost_factor = torch.exp(-(diff * 10) ** 2)  # Higher for dim stars (small diff)
        boost_amount = enhancement_params * boost_factor * star_map
        enhanced = x + boost_amount * 0.7  # Control the enhancement strength
        enhanced = torch.clamp(enhanced, 0, 1)
        
        # Refine the output by combining original, enhanced, and star map
        output = self.output_refine(torch.cat([x, enhanced, star_map], dim=1))
        
        return output, star_map


class ImprovedAstroEnhancement(nn.Module):
    """
    Complete enhancement pipeline for astronomical images.
    Specifically designed to enhance dim stars for better detection.
    """
    def __init__(self, in_channels=3):
        super(ImprovedAstroEnhancement, self).__init__()
        
        # Star enhancement module
        self.enhancement_module = ImprovedStarEnhancementModule(in_channels)
        
        # Fusion module for combining original and enhanced images
        self.fusion_module = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Store original input
        original = x
        
        # Apply the enhanced star module
        enhanced, star_map = self.enhancement_module(x)
        
        # Fuse original and enhanced images
        output = self.fusion_module(torch.cat([original, enhanced, star_map], dim=1))
        
        # For visualization/debugging during training
        # if self.training:
        #     return output, enhanced, star_map
        # else:
        return output


# Utility functions for visualizing the enhancement

def visualize_enhancement(model, image_tensor):
    """
    Utility function to visualize the enhancement process.
    
    Args:
        model: The enhancement model
        image_tensor: Input image tensor (B, C, H, W)
        
    Returns:
        Dictionary of tensors for visualization
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, ImprovedAstroEnhancement) and model.training:
            output, enhanced, star_map = model(image_tensor)
            return {
                'original': image_tensor,
                'enhanced_only': enhanced,
                'star_map': star_map,
                'final_output': output
            }
        elif isinstance(model, ImprovedStarEnhancementModule):
            output, star_map = model(image_tensor)
            return {
                'original': image_tensor,
                'enhanced': output,
                'star_map': star_map
            }
        else:
            output = model(image_tensor)
            return {
                'original': image_tensor,
                'final_output': output
            }