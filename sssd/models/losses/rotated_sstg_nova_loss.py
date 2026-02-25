#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_LOSSES
from .utils.sample_tools import xywha2rbox
from .utils.ot_tools import OT_Loss

@ROTATED_LOSSES.register_module()
class RotatedSingleStageDTLoss(nn.Module):
    def __init__(self, cls_channels=16, loss_type='pr_origin_p5', cls_loss_type='bce',
                 aux_loss=None, sigma_scale=0.5, rbox_pts_ratio=0.25, aux_loss_cfg=dict(),
                 dynamic_weight='ang', dynamic_fix_weight=None, co_mining_weight=0.7,
                 low_conf_threshold=0.15, low_conf_ratio=0.2, low_conf_weight=0.5):
        """
        Symmetry Aware Single Stage Dense Teacher Loss with Co-mining.
        Args:
            cls_channels:
            loss_type:
            aux_loss (Optional | str): additional loss for auxiliary
            co_mining_weight: 控制co-mining损失的权重
            low_conf_threshold: 低置信度区域的阈值
            low_conf_ratio: 低置信度样本占总样本的比例
            low_conf_weight: 低置信度样本的损失权重
        """
        super(RotatedSingleStageDTLoss, self).__init__()
        self.cls_channels = cls_channels
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')

        self.loss_type = loss_type
        assert cls_loss_type in ['bce']
        self.cls_loss_type = cls_loss_type
        if aux_loss:
            assert aux_loss in ['ot_loss_norm', 'ot_ang_loss_norm']
            self.ot_weight = aux_loss_cfg.pop('loss_weight', 1.)
            self.cost_type = aux_loss_cfg.pop('cost_type', 'all')
            assert self.cost_type in ['all', 'dist', 'score']
            self.clamp_ot = aux_loss_cfg.pop('clamp_ot', False)
            self.gc_loss = OT_Loss(**aux_loss_cfg)
        self.aux_loss = aux_loss
        self.apply_ot = self.aux_loss
        self.sigma_scale = sigma_scale
        self.rbox_pts_ratio = rbox_pts_ratio
        assert dynamic_weight in ['ang', '10ang', '50ang', '100ang']
        self.dynamic_weight = dynamic_weight
        if dynamic_fix_weight:
            self.dynamic_fix_weight = dynamic_fix_weight
        else:
            if self.dynamic_weight == 'ang':
                self.dynamic_fix_weight = 1.0
            else:
                self.dynamic_fix_weight = 1.0
        
        # Co-mining相关参数
        self.co_mining_weight = co_mining_weight
        
        # 低置信度伪标签相关参数
        self.low_conf_threshold = low_conf_threshold
        self.low_conf_ratio = low_conf_ratio
        self.low_conf_weight = low_conf_weight

    def forward(self, teacher_logits, teacher_logits_strong, student_logits, 
                student_logits_weak=None, ratio=0.01, teacher_preds=None, 
                teacher_preds_strong=None, img_metas=None):
        """
        Args:
            teacher_logits (Tuple): 弱增强视图上的教师模型输出
            teacher_logits_strong (Tuple): 强增强视图上的教师模型输出
            student_logits (Tuple): 学生模型在强增强视图上的输出
            student_logits_weak (Tuple): 学生模型在弱增强视图上的输出 (用于co-mining)
            ratio (float): 采样比例
            teacher_preds (Optional | List): 弱增强教师模型的预测结果
            teacher_preds_strong (Optional | List): 强增强教师模型的预测结果 (用于co-mining)
            img_metas (Optional | Dict): 图像元信息

        Returns:
            dict: 损失字典
        """

        gpu_device = teacher_logits[0][0].device
        
        if self.loss_type in ['pr_origin_p5']:
            # 主要分支：弱增强教师为强增强学生生成伪标签
            mask = torch.zeros((len(teacher_preds), 1, 1024, 1024), device=gpu_device)
            low_conf_mask = torch.zeros((len(teacher_preds), 1, 1024, 1024), device=gpu_device)
            
            # Co-mining分支：强增强教师为弱增强学生生成伪标签
            mask_co_mining = torch.zeros((len(teacher_preds), 1, 1024, 1024), device=gpu_device)
            
            # 处理弱增强教师预测（主要分支）
            for img_idx, bbox_result in enumerate(teacher_preds):
                bboxes = []
                low_conf_bboxes = []
                
                for cls_idx, bbox_per_cls in enumerate(bbox_result):
                    if bbox_per_cls.shape[0] > 0:
                        # 高置信度边界框
                        high_conf_idx = bbox_per_cls[:, -2] > 0.5
                        if high_conf_idx.any():
                            bboxes.append(
                                np.hstack([bbox_per_cls[high_conf_idx], cls_idx * np.ones((high_conf_idx.sum(), 1))])
                            )
                        
                        # 低置信度边界框
                        low_conf_idx = (bbox_per_cls[:, -2] > self.low_conf_threshold) & (bbox_per_cls[:, -2] <= 0.5)
                        if low_conf_idx.any():
                            low_conf_bboxes.append(
                                np.hstack([bbox_per_cls[low_conf_idx], cls_idx * np.ones((low_conf_idx.sum(), 1))])
                            )
                
                # 处理高置信度边界框
                if len(bboxes) > 0:
                    bboxes = np.vstack(bboxes)
                    mask[img_idx] = xywha2rbox(bboxes, gpu_device,
                                           img_meta=img_metas['unsup_strong'][img_idx],
                                           ratio=self.rbox_pts_ratio).to(gpu_device)
                
                # 处理低置信度边界框 - 添加额外检查
                if len(low_conf_bboxes) > 0:
                    low_conf_bboxes = np.vstack(low_conf_bboxes)
                    
                    # 如果高置信度样本存在且低置信度样本过多，进行采样
                    if len(bboxes) > 0 and len(low_conf_bboxes) > len(bboxes) * self.low_conf_ratio:
                        sample_indices = np.random.choice(
                            len(low_conf_bboxes), 
                            int(len(bboxes) * self.low_conf_ratio), 
                            replace=False
                        )
                        low_conf_bboxes = low_conf_bboxes[sample_indices]
                    
                    # 确保采样后仍有数据
                    if len(low_conf_bboxes) > 0:
                        low_conf_mask[img_idx] = xywha2rbox(low_conf_bboxes, gpu_device,
                                                   img_meta=img_metas['unsup_strong'][img_idx],
                                                   ratio=self.rbox_pts_ratio).to(gpu_device)
            
            # 处理强增强教师预测（Co-mining分支）
            if teacher_preds_strong is not None:
                for img_idx, bbox_result in enumerate(teacher_preds_strong):
                    bboxes = []
                    
                    for cls_idx, bbox_per_cls in enumerate(bbox_result):
                        if bbox_per_cls.shape[0] > 0:
                            # 只使用高置信度边界框
                            high_conf_idx = bbox_per_cls[:, -2] > 0.5
                            if high_conf_idx.any():
                                bboxes.append(
                                    np.hstack([bbox_per_cls[high_conf_idx], cls_idx * np.ones((high_conf_idx.sum(), 1))])
                                )
                    
                    # 确保有边界框数据才调用xywha2rbox
                    if len(bboxes) > 0:
                        bboxes = np.vstack(bboxes)
                        # 确保bboxes不为空
                        if bboxes.shape[0] > 0:
                            mask_co_mining[img_idx] = xywha2rbox(bboxes, gpu_device,
                                                   img_meta=img_metas['unsup_weak'][img_idx],  # 注意：这里应该用弱增强的meta
                                                   ratio=self.rbox_pts_ratio).to(gpu_device)
            
            # 调整mask大小
            mask = F.interpolate(mask.float(), (128, 128)).bool().squeeze(1)
            low_conf_mask = F.interpolate(low_conf_mask.float(), (128, 128)).bool().squeeze(1)
            mask_co_mining = F.interpolate(mask_co_mining.float(), (128, 128)).bool().squeeze(1)
            
            # 计算有效样本数
            num_valid = mask.sum()
            num_low_conf_valid = low_conf_mask.sum()
            num_valid_co_mining = mask_co_mining.sum()
                
            # 如果没有有效样本，直接返回零损失
            if num_valid == 0 and num_low_conf_valid == 0 and num_valid_co_mining == 0:
                zero_loss = torch.tensor(0., device=gpu_device)
                loss_dict = dict(
                    loss_raw=zero_loss,
                    loss_co_mining=zero_loss,
                    loss_low_conf=zero_loss
                )
                if self.apply_ot:
                    loss_dict.update(loss_gc=zero_loss)
                return loss_dict
            
            # 提取模型输出
            t_cls_scores, t_bbox_preds, t_angle_preds, t_centernesses = teacher_logits
            s_cls_scores, s_bbox_preds, s_angle_preds, s_centernesses = student_logits
            t_cls_scores_strong, t_bbox_preds_strong, t_angle_preds_strong, t_centernesses_strong = teacher_logits_strong
            
            target_size = (128, 128)
            
            # ============= 主要分支：弱增强教师 -> 强增强学生 =============
            loss_raw = torch.tensor(0., device=gpu_device)
            if num_valid > 0:
                # 提取主要分支的特征
                t_cls_main = F.interpolate(t_cls_scores[0], target_size).permute(0, 2, 3, 1)[mask]
                t_bbox_main = F.interpolate(t_bbox_preds[0], target_size).permute(0, 2, 3, 1)[mask]
                t_angle_main = F.interpolate(t_angle_preds[0], target_size).permute(0, 2, 3, 1)[mask]
                t_centerness_main = F.interpolate(t_centernesses[0], target_size).permute(0, 2, 3, 1)[mask]

                s_cls_main = F.interpolate(s_cls_scores[0], target_size).permute(0, 2, 3, 1)[mask]
                s_bbox_main = F.interpolate(s_bbox_preds[0], target_size).permute(0, 2, 3, 1)[mask]
                s_angle_main = F.interpolate(s_angle_preds[0], target_size).permute(0, 2, 3, 1)[mask]
                s_centerness_main = F.interpolate(s_centernesses[0], target_size).permute(0, 2, 3, 1)[mask]

                t_bbox_main = torch.cat([t_bbox_main, t_angle_main], dim=-1)
                s_bbox_main = torch.cat([s_bbox_main, s_angle_main], dim=-1)

                # 计算动态权重
                with torch.no_grad():
                    loss_weight = torch.abs(t_bbox_main[:, -1] - s_bbox_main[:, -1]) / np.pi
                    if self.dynamic_weight == '10ang':
                        loss_weight = torch.clamp(10 * loss_weight.unsqueeze(-1), 0, 1) + 1
                    elif self.dynamic_weight == '50ang':
                        loss_weight = torch.clamp(50 * loss_weight.unsqueeze(-1), 0, 1) + 1
                    elif self.dynamic_weight == '100ang':
                        loss_weight = torch.clamp(100 * loss_weight.unsqueeze(-1), 0, 1) + 1
                    else:
                        loss_weight = loss_weight.unsqueeze(-1) + 1

                # 计算主要损失
                if self.cls_loss_type == 'bce':
                    loss_cls = F.binary_cross_entropy(
                        s_cls_main.sigmoid(),
                        t_cls_main.sigmoid(),
                        reduction="none",
                    )
                    loss_cls = (loss_cls * loss_weight).sum() / (num_valid + 1e-10)

                loss_bbox = self.bbox_loss(s_bbox_main, t_bbox_main) * t_centerness_main.sigmoid()
                loss_bbox = (loss_bbox * loss_weight).sum() / (num_valid + 1e-10)

                loss_centerness = F.binary_cross_entropy(
                    s_centerness_main.sigmoid(),
                    t_centerness_main.sigmoid(),
                    reduction='none' 
                )
                loss_centerness = (loss_centerness * loss_weight).sum() / (num_valid + 1e-10)

                loss_raw = loss_cls + loss_bbox + loss_centerness

            # ============= Co-mining分支：强增强教师 -> 弱增强学生 =============
            loss_co_mining = torch.tensor(0., device=gpu_device)
            if num_valid_co_mining > 0 and student_logits_weak is not None:
                s_cls_scores_weak, s_bbox_preds_weak, s_angle_preds_weak, s_centernesses_weak = student_logits_weak
                
                # 提取Co-mining分支的特征
                t_cls_co = F.interpolate(t_cls_scores_strong[0], target_size).permute(0, 2, 3, 1)[mask_co_mining]
                t_bbox_co = F.interpolate(t_bbox_preds_strong[0], target_size).permute(0, 2, 3, 1)[mask_co_mining]
                t_angle_co = F.interpolate(t_angle_preds_strong[0], target_size).permute(0, 2, 3, 1)[mask_co_mining]
                t_centerness_co = F.interpolate(t_centernesses_strong[0], target_size).permute(0, 2, 3, 1)[mask_co_mining]

                s_cls_co = F.interpolate(s_cls_scores_weak[0], target_size).permute(0, 2, 3, 1)[mask_co_mining]
                s_bbox_co = F.interpolate(s_bbox_preds_weak[0], target_size).permute(0, 2, 3, 1)[mask_co_mining]
                s_angle_co = F.interpolate(s_angle_preds_weak[0], target_size).permute(0, 2, 3, 1)[mask_co_mining]
                s_centerness_co = F.interpolate(s_centernesses_weak[0], target_size).permute(0, 2, 3, 1)[mask_co_mining]

                t_bbox_co = torch.cat([t_bbox_co, t_angle_co], dim=-1)
                s_bbox_co = torch.cat([s_bbox_co, s_angle_co], dim=-1)

                # 计算Co-mining动态权重
                with torch.no_grad():
                    loss_weight_co = torch.abs(t_bbox_co[:, -1] - s_bbox_co[:, -1]) / np.pi
                    if self.dynamic_weight == '10ang':
                        loss_weight_co = torch.clamp(10 * loss_weight_co.unsqueeze(-1), 0, 1) + 1
                    elif self.dynamic_weight == '50ang':
                        loss_weight_co = torch.clamp(50 * loss_weight_co.unsqueeze(-1), 0, 1) + 1
                    elif self.dynamic_weight == '100ang':
                        loss_weight_co = torch.clamp(100 * loss_weight_co.unsqueeze(-1), 0, 1) + 1
                    else:
                        loss_weight_co = loss_weight_co.unsqueeze(-1) + 1

                # 计算Co-mining损失
                if self.cls_loss_type == 'bce':
                    loss_cls_co = F.binary_cross_entropy(
                        s_cls_co.sigmoid(),
                        t_cls_co.sigmoid(),
                        reduction="none",
                    )
                    loss_cls_co = (loss_cls_co * loss_weight_co).sum() / (num_valid_co_mining + 1e-10)

                loss_bbox_co = self.bbox_loss(s_bbox_co, t_bbox_co) * t_centerness_co.sigmoid()
                loss_bbox_co = (loss_bbox_co * loss_weight_co).sum() / (num_valid_co_mining + 1e-10)

                loss_centerness_co = F.binary_cross_entropy(
                    s_centerness_co.sigmoid(),
                    t_centerness_co.sigmoid(),
                    reduction='none' 
                )
                loss_centerness_co = (loss_centerness_co * loss_weight_co).sum() / (num_valid_co_mining + 1e-10)

                loss_co_mining = loss_cls_co + loss_bbox_co + loss_centerness_co

            # 构建损失字典
            unsup_losses = dict(
                loss_raw=self.dynamic_fix_weight * loss_raw,
                loss_co_mining=self.co_mining_weight * loss_co_mining
            )

            # ============= 辅助损失（OT损失）- 添加数值稳定性处理 =============
            if self.aux_loss and num_valid > 0:
                loss_gc_total = torch.zeros(1).to(gpu_device)
                
                if self.loss_type in ['pr_origin_p5']:
                    if self.aux_loss in ['ot_ang_loss_norm']:
                        t_score_map = teacher_logits[2][0]
                        s_score_map = student_logits[2][0]
                    else:
                        t_score_map = teacher_logits[0][0]
                        s_score_map = student_logits[0][0]
                    
                    batch_size = t_score_map.shape[0]
                    
                    if teacher_logits[0][0].shape[-2:] != mask.shape[-2:]:
                        t_score_map = F.interpolate(t_score_map, mask.shape[-2:]).permute(0, 2, 3, 1)
                        s_score_map = F.interpolate(s_score_map, mask.shape[-2:]).permute(0, 2, 3, 1)
                    else:
                        t_score_map = t_score_map.permute(0, 2, 3, 1)
                        s_score_map = s_score_map.permute(0, 2, 3, 1)

                    # 数值稳定性处理
                    if self.aux_loss in ['ot_loss_norm']:
                        # 添加温度参数防止数值不稳定
                        temperature = 1.0
                        t_score_map = torch.softmax(t_score_map / temperature, dim=-1)
                        s_score_map = torch.softmax(s_score_map / temperature, dim=-1)
                        # 防止接近0的值
                        t_score_map = torch.clamp(t_score_map, min=1e-8, max=1.0)
                        s_score_map = torch.clamp(s_score_map, min=1e-8, max=1.0)
                    elif self.aux_loss in ['ot_ang_loss_norm']:
                        t_score_map = torch.abs(t_score_map) / np.pi
                        s_score_map = torch.abs(s_score_map) / np.pi
                        # 限制范围防止数值不稳定
                        t_score_map = torch.clamp(t_score_map, min=1e-6, max=1.0)
                        s_score_map = torch.clamp(s_score_map, min=1e-6, max=1.0)
                    
                    valid_ot_count = 0
                    for img_idx in range(batch_size):
                        if mask[img_idx].any():
                            t_score, score_cls = torch.max(t_score_map[img_idx][mask[img_idx]], dim=-1)
                            s_score = s_score_map[img_idx][mask[img_idx]][
                                torch.arange(t_score.shape[0], device=gpu_device),
                                score_cls]
                            pts = mask[img_idx].nonzero()
                            
                            # 检查是否有足够的点
                            if len(pts) <= 1:
                                continue
                            
                            # 检查数值有效性
                            if torch.isnan(t_score).any() or torch.isnan(s_score).any():
                                continue
                            
                            if torch.isinf(t_score).any() or torch.isinf(s_score).any():
                                continue
                            
                            # 检查分数范围
                            if (t_score < 1e-8).any() or (s_score < 1e-8).any():
                                continue
                                
                            try:
                                # 添加更严格的clamp_ot设置
                                ot_loss_val = self.gc_loss(t_score, s_score, pts, 
                                                         cost_type=self.cost_type,
                                                         clamp_ot=True)  # 强制启用clamp
                                
                                # 检查OT损失是否有效
                                if torch.isfinite(ot_loss_val) and ot_loss_val >= 0:
                                    # 限制OT损失的最大值防止爆炸
                                    ot_loss_val = torch.clamp(ot_loss_val, max=10.0)
                                    loss_gc_total += ot_loss_val
                                    valid_ot_count += 1
                                    
                            except Exception as e:
                                # 静默处理异常，避免训练中断
                                continue
                    
                    # 只有当有有效的OT损失时才更新
                    if valid_ot_count > 0:
                        unsup_losses.update(loss_gc=self.ot_weight * loss_gc_total.sum() / valid_ot_count)
                    else:
                        unsup_losses.update(loss_gc=torch.tensor(0., device=gpu_device))
                else:
                    raise RuntimeError(f"Not support {self.loss_type}")
            
            elif self.apply_ot:
                unsup_losses.update(loss_gc=torch.tensor(0., device=gpu_device))
                
                        # ============= 低置信度损失 =============
            loss_low_conf = torch.tensor(0., device=gpu_device)
            if num_low_conf_valid > 0:
                # 提取低置信度区域的特征
                s_cls_scores_low = F.interpolate(s_cls_scores[0], target_size).permute(0, 2, 3, 1)[low_conf_mask]
                s_bbox_preds_low = F.interpolate(s_bbox_preds[0], target_size).permute(0, 2, 3, 1)[low_conf_mask]
                s_angle_preds_low = F.interpolate(s_angle_preds[0], target_size).permute(0, 2, 3, 1)[low_conf_mask]
                s_centernesses_low = F.interpolate(s_centernesses[0], target_size).permute(0, 2, 3, 1)[low_conf_mask]
                
                t_cls_scores_low = F.interpolate(t_cls_scores[0], target_size).permute(0, 2, 3, 1)[low_conf_mask]
                t_bbox_preds_low = F.interpolate(t_bbox_preds[0], target_size).permute(0, 2, 3, 1)[low_conf_mask]
                t_angle_preds_low = F.interpolate(t_angle_preds[0], target_size).permute(0, 2, 3, 1)[low_conf_mask]
                t_centernesses_low = F.interpolate(t_centernesses[0], target_size).permute(0, 2, 3, 1)[low_conf_mask]
                
                s_bbox_preds_low = torch.cat([s_bbox_preds_low, s_angle_preds_low], dim=-1)
                t_bbox_preds_low = torch.cat([t_bbox_preds_low, t_angle_preds_low], dim=-1)
                
                # 计算低置信度样本的损失
                if self.cls_loss_type == 'bce':
                    loss_cls_low = F.binary_cross_entropy(
                        s_cls_scores_low.sigmoid(),
                        t_cls_scores_low.sigmoid(),
                        reduction="none",
                    )
                    loss_cls_low = loss_cls_low.sum() / (num_low_conf_valid + 1e-10)
                else:
                    raise RuntimeError(f"Not support {self.cls_loss_type}")
                
                loss_bbox_low = self.bbox_loss(
                    s_bbox_preds_low,
                    t_bbox_preds_low,
                ) * t_centernesses_low.sigmoid()
                loss_bbox_low = loss_bbox_low.sum() / (num_low_conf_valid + 1e-10)
                
                loss_centerness_low = F.binary_cross_entropy(
                    s_centernesses_low.sigmoid(),
                    t_centernesses_low.sigmoid(),
                    reduction='none' 
                )
                loss_centerness_low = loss_centerness_low.sum() / (num_low_conf_valid + 1e-10)
                
                # 合并低置信度损失
                loss_low_conf = loss_cls_low + loss_bbox_low + loss_centerness_low

            # 更新损失字典
            unsup_losses.update(loss_low_conf=self.low_conf_weight * loss_low_conf)
            
            return unsup_losses