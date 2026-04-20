import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pandas as pd
from scipy.optimize import linear_sum_assignment

@dataclass
class Star:
    """恒星数据类"""
    x_c: float
    y_c: float
    a: float  # 半长轴
    b: float  # 半短轴
    theta: float  # 旋转角度（弧度）
    snr: float = None  # 信噪比
    score: float = 1.0  # 检测置信度
    
class StarDetectionEvaluator:
    def __init__(self, iou_threshold=0.1, distance_threshold=None):
        """
        初始化评估器
        
        Args:
            iou_threshold: IoU阈值，用于判断检测是否匹配
            distance_threshold: 距离阈值（像素），如果设置则使用距离匹配而非IoU
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        
    def ellipse_to_rotated_bbox(self, star: Star) -> np.ndarray:
        """
        将椭圆参数转换为旋转矩形框（DOTA格式）
        使用2倍的椭圆轴长作为矩形的宽高，以更好地包围椭圆
        """
        x_c, y_c, a, b, theta = star.x_c, star.y_c, star.a, star.b, star.theta
        
        # 使用2倍轴长确保矩形完全包围椭圆
        half_width = a * 2
        half_height = b * 2
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # 四个角点在局部坐标系中的位置
        corners_local = np.array([
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height]
        ])
        
        # 旋转矩阵
        rotation_matrix = np.array([
            [cos_t, -sin_t],
            [sin_t, cos_t]
        ])
        
        # 旋转并平移到世界坐标系
        corners_world = np.dot(corners_local, rotation_matrix.T) + np.array([x_c, y_c])
        
        return corners_world.flatten()
    
    def calculate_ellipse_iou(self, star1: Star, star2: Star) -> float:
        """
        计算两个椭圆的IoU（使用近似方法）
        """
        # 方法1：使用距离和尺寸相似性的组合度量
        center_dist = np.sqrt((star1.x_c - star2.x_c)**2 + (star1.y_c - star2.y_c)**2)
        
        # 平均半径
        r1 = (star1.a + star1.b) / 2
        r2 = (star2.a + star2.b) / 2
        
        # 如果中心距离大于两个椭圆半径之和，则没有重叠
        if center_dist > (r1 + r2):
            return 0.0
        
        # 简化的IoU估计
        if center_dist < abs(r1 - r2):
            # 一个椭圆包含另一个
            return min(r1, r2)**2 / max(r1, r2)**2
        else:
            # 部分重叠，使用简化公式
            overlap = max(0, r1 + r2 - center_dist) / (r1 + r2)
            return overlap
    
    def calculate_iou_rotated_bbox(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        计算两个旋转矩形框的IoU
        """
        try:
            from shapely.geometry import Polygon
            from shapely.errors import TopologicalError
            
            # 将扁平数组转换为点坐标
            points1 = bbox1.reshape(4, 2)
            points2 = bbox2.reshape(4, 2)
            
            # 创建多边形
            poly1 = Polygon(points1)
            poly2 = Polygon(points2)
            
            # 确保多边形有效
            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)
            
            try:
                # 计算交集和并集
                intersection = poly1.intersection(poly2).area
                union = poly1.union(poly2).area
                
                if union == 0:
                    return 0.0
                
                return intersection / union
            except TopologicalError:
                # 如果拓扑错误，返回0
                return 0.0
                
        except ImportError:
            print("警告: shapely未安装，使用简化的IoU计算")
            return self.simple_box_iou(bbox1, bbox2)
    
    def simple_box_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        简化的边界框IoU计算（使用轴对齐边界框）
        """
        points1 = bbox1.reshape(4, 2)
        points2 = bbox2.reshape(4, 2)
        
        # 获取轴对齐边界框
        x1_min, y1_min = points1.min(axis=0)
        x1_max, y1_max = points1.max(axis=0)
        x2_min, y2_min = points2.min(axis=0)
        x2_max, y2_max = points2.max(axis=0)
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def calculate_distance(self, star1: Star, star2: Star) -> float:
        """计算两个恒星中心的欧氏距离"""
        return np.sqrt((star1.x_c - star2.x_c)**2 + (star1.y_c - star2.y_c)**2)
    
    def load_annotations(self, ann_file: str, snr_file: str = None) -> List[Star]:
        """
        加载注释文件和SNR文件
        """
        stars = []
        
        if not os.path.exists(ann_file):
            return stars
        
        # 加载注释
        with open(ann_file, 'r') as f:
            annotations = f.readlines()
        
        # 加载SNR（如果提供）
        snrs = None
        if snr_file and os.path.exists(snr_file):
            with open(snr_file, 'r') as f:
                snr_lines = f.readlines()
                snrs = []
                for line in snr_lines:
                    try:
                        snrs.append(float(line.strip()))
                    except ValueError:
                        snrs.append(None)
        
        # 创建Star对象
        for i, line in enumerate(annotations):
            line = line.strip()
            if not line:
                continue
            
            params = line.split()
            if len(params) >= 5:
                try:
                    star = Star(
                        x_c=float(params[0]),
                        y_c=float(params[1]),
                        a=float(params[2]),
                        b=float(params[3]),
                        theta=float(params[4]),
                        snr=snrs[i] if snrs and i < len(snrs) else None,
                        score=float(params[5]) if len(params) > 5 else 1.0
                    )
                    stars.append(star)
                except ValueError:
                    print(f"警告：无法解析行 {i+1} 在文件 {ann_file}")
                    continue
        
        return stars
    
    def match_detections_hungarian(self, gt_stars: List[Star], pred_stars: List[Star]) -> Tuple[List[int], List[int], List[float]]:
        """
        使用匈牙利算法进行最优匹配
        
        Returns:
            matched_gt_indices: 匹配的GT索引
            matched_pred_indices: 匹配的预测索引  
            ious: 对应的IoU值
        """
        if not gt_stars or not pred_stars:
            return [], [], []
        
        n_gt = len(gt_stars)
        n_pred = len(pred_stars)
        
        # 构建代价矩阵
        cost_matrix = np.zeros((n_gt, n_pred))
        
        for i, gt_star in enumerate(gt_stars):
            for j, pred_star in enumerate(pred_stars):
                if self.distance_threshold is not None:
                    # 使用距离匹配
                    distance = self.calculate_distance(gt_star, pred_star)
                    if distance <= self.distance_threshold:
                        cost_matrix[i, j] = 1.0 - (distance / self.distance_threshold)
                    else:
                        cost_matrix[i, j] = 0
                else:
                    # 使用IoU匹配
                    iou = self.calculate_ellipse_iou(gt_star, pred_star)
                    cost_matrix[i, j] = iou
        
        # 使用匈牙利算法找到最优匹配
        # 注意：linear_sum_assignment最小化代价，所以我们使用负IoU
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        
        # 过滤掉IoU/相似度低于阈值的匹配
        matched_gt = []
        matched_pred = []
        ious = []
        
        threshold = self.iou_threshold if self.distance_threshold is None else 0.01
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] >= threshold:
                matched_gt.append(i)
                matched_pred.append(j)
                ious.append(cost_matrix[i, j])
        
        return matched_gt, matched_pred, ious
    
    def evaluate_single_image(self, gt_stars: List[Star], pred_stars: List[Star]) -> Dict:
        """
        评估单张图片
        """
        # 按置信度排序预测（如果有置信度分数）
        pred_stars = sorted(pred_stars, key=lambda x: x.score, reverse=True)
        
        # 匹配检测
        matched_gt, matched_pred, ious = self.match_detections_hungarian(gt_stars, pred_stars)
        
        # 计算TP, FP, FN
        tp = len(matched_gt)
        fp = len(pred_stars) - tp
        fn = len(gt_stars) - tp
        
        # 计算精度和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'matched_gt': matched_gt,
            'matched_pred': matched_pred,
            'mean_iou': np.mean(ious) if ious else 0
        }
    
    def evaluate_by_snr_ranges(self, 
                               gt_folder: str,
                               pred_folder: str,
                               snr_folder: str,
                               snr_ranges: List[Tuple[float, float]] = None,
                               use_distance_matching: bool = False,
                               distance_threshold: float = 3.0) -> pd.DataFrame:
        """
        按SNR范围评估检测结果
        
        Args:
            gt_folder: GT注释文件夹
            pred_folder: 预测注释文件夹
            snr_folder: SNR文件夹
            snr_ranges: SNR范围列表
            use_distance_matching: 是否使用距离匹配而非IoU
            distance_threshold: 距离阈值（像素）
        
        Returns:
            评估结果DataFrame
        """
        if use_distance_matching:
            self.distance_threshold = distance_threshold
            self.iou_threshold = None
        
        if snr_ranges is None:
            snr_ranges = [(0, 5), (5, 10), (10, 20), (20, 50), (50, float('inf'))]
        
        results = []
        
        # 获取所有图片文件
        gt_files = sorted(Path(gt_folder).glob('*.txt'))
        
        if not gt_files:
            print(f"警告：在 {gt_folder} 中未找到任何txt文件")
            return pd.DataFrame()
        
        print(f"找到 {len(gt_files)} 个GT文件")
        
        # 按SNR范围统计
        for snr_min, snr_max in snr_ranges:
            range_results = {
                'snr_range': f'{snr_min}-{snr_max if snr_max != float("inf") else "∞"}',
                'total_gt': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'images_processed': 0
            }
            
            for gt_file in gt_files:
                # 构建对应的预测文件和SNR文件路径
                base_name = gt_file.stem
                pred_file = Path(pred_folder) / f'{base_name}.png.txt'
                snr_file = Path(snr_folder) / f'{base_name}.txt'
                
                if not pred_file.exists():
                    print(f"警告: 预测文件 {pred_file} 不存在")
                    continue
                
                # 加载数据
                gt_stars_all = self.load_annotations(str(gt_file), str(snr_file))
                pred_stars_all = self.load_annotations(str(pred_file))
                
                # 筛选SNR范围内的GT
                gt_stars_in_range = [
                    star for star in gt_stars_all 
                    if star.snr is not None and snr_min <= star.snr < snr_max
                ]
                
                if not gt_stars_in_range:
                    continue
                
                # 对于每个SNR范围内的GT，需要在所有预测中查找匹配
                # 这里是关键：我们评估的是对特定SNR范围内GT的检测能力
                eval_result = self.evaluate_single_image(gt_stars_in_range, pred_stars_all)
                
                # 累积结果
                range_results['total_gt'] += len(gt_stars_in_range)
                range_results['tp'] += eval_result['tp']
                # 注意：FP应该只计算与该SNR范围GT相关的误检
                # 这里简化处理，不重复计算全局FP
                range_results['fn'] += eval_result['fn']
                range_results['images_processed'] += 1
            
            # 计算该SNR范围内所有图片的FP
            # FP的计算比较复杂，这里采用保守估计
            range_results['fp'] = 0  # 暂时设为0，因为跨SNR范围的FP难以准确计算
            
            # 计算总体指标
            tp = range_results['tp']
            fn = range_results['fn']
            
            # Recall是最重要的指标
            range_results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Precision需要特殊处理，因为FP的定义在分范围评估时比较复杂
            # 这里我们只报告recall
            range_results['precision'] = None  # 暂不计算
            
            results.append(range_results)
        
        df = pd.DataFrame(results)
        
        # 添加总体统计
        total_row = {
            'snr_range': 'Overall',
            'total_gt': df['total_gt'].sum(),
            'tp': df['tp'].sum(),
            'fn': df['fn'].sum(),
            'recall': df['tp'].sum() / (df['tp'].sum() + df['fn'].sum()) if (df['tp'].sum() + df['fn'].sum()) > 0 else 0,
            'images_processed': df['images_processed'].max()
        }
        
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        return df
    
    def evaluate_with_global_metrics(self, 
                                    gt_folder: str,
                                    pred_folder: str,
                                    snr_folder: str,
                                    snr_ranges: List[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        更准确的评估方法，同时计算Precision和Recall
        """
        if snr_ranges is None:
            snr_ranges = [(0, 5), (5, 10), (10, 20), (20, 50), (50, float('inf'))]
        
        # 存储每个范围的所有匹配结果
        range_data = {str(r): {'gt': [], 'pred': [], 'matched': []} for r in snr_ranges}
        
        gt_files = sorted(Path(gt_folder).glob('*.txt'))
        
        for gt_file in gt_files:
            base_name = gt_file.stem
            pred_file = Path(pred_folder) / f'{base_name}.png.txt'
            snr_file = Path(snr_folder) / f'{base_name}.txt'
            
            if not pred_file.exists():
                continue
            
            # 加载所有数据
            gt_stars = self.load_annotations(str(gt_file), str(snr_file))
            pred_stars = self.load_annotations(str(pred_file))
            
            # 进行全局匹配
            matched_gt, matched_pred, _ = self.match_detections_hungarian(gt_stars, pred_stars)
            
            # 按SNR分配结果
            for i, gt_star in enumerate(gt_stars):
                if gt_star.snr is None:
                    continue
                
                for snr_min, snr_max in snr_ranges:
                    if snr_min <= gt_star.snr < snr_max:
                        range_key = f'{snr_min}-{snr_max if snr_max != float("inf") else "∞"}'
                        range_data[range_key]['gt'].append(i)
                        
                        # 检查这个GT是否被匹配
                        if i in matched_gt:
                            pred_idx = matched_pred[matched_gt.index(i)]
                            range_data[range_key]['matched'].append((i, pred_idx))
                        break
        
        # 计算每个范围的指标
        results = []
        for snr_min, snr_max in snr_ranges:
            range_key = f'{snr_min}-{snr_max if snr_max != float("inf") else "∞"}'
            
            total_gt = len(range_data[range_key]['gt'])
            tp = len(range_data[range_key]['matched'])
            fn = total_gt - tp
            
            result = {
                'snr_range': range_key,
                'total_gt': total_gt,
                'tp': tp,
                'fn': fn,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_results(self, results_df: pd.DataFrame, save_path: str = None):
        """
        可视化评估结果（专注于Recall）
        """
        # 过滤掉Overall行
        plot_df = results_df[results_df['snr_range'] != 'Overall'].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Recall vs SNR
        axes[0].bar(plot_df['snr_range'], plot_df['recall'])
        axes[0].set_xlabel('SNR Range')
        axes[0].set_ylabel('Recall')
        axes[0].set_title('Recall by SNR Range')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # 在条形图上添加数值
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            axes[0].text(i, row['recall'], f"{row['recall']:.3f}", 
                        ha='center', va='bottom')
        
        # Detection Statistics
        x = np.arange(len(plot_df))
        width = 0.35
        axes[1].bar(x - width/2, plot_df['tp'], width, label='TP (Detected)', color='green')
        axes[1].bar(x + width/2, plot_df['fn'], width, label='FN (Missed)', color='red')
        axes[1].set_xlabel('SNR Range')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Detection Statistics by SNR Range')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(plot_df['snr_range'], rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# 使用示例
def main():
    """主函数：演示如何使用评估器"""
    
    # 设置路径
    gt_folder = '/home/flh/datasets/LAMOST_new/dataset_ori/test/gt_norm/'  # GT注释文件夹
    pred_folder = '/home/flh/projects/focal-teacher/work_dir/lamost/fcoal_teacher/sparse/sparse_50/test_img/iter_115000/det/'  # 预测注释文件夹
    snr_folder = '/home/flh/datasets/LAMOST_new/dataset_ori/test/snr_txt_mutil/'    # SNR文件夹
    
    # 方法1：使用IoU匹配（适用于有准确边界框的情况）
    print("=" * 80)
    print("方法1: 使用IoU匹配")
    print("=" * 80)
    
    evaluator_iou = StarDetectionEvaluator(iou_threshold=0.3)  # 降低IoU阈值
    
    results_iou = evaluator_iou.evaluate_by_snr_ranges(
        gt_folder=gt_folder,
        pred_folder=pred_folder,
        snr_folder=snr_folder,
        snr_ranges=[(0, 5), (5, 10), (10, 20), (20, 50), (50, float('inf'))]
    )
    
    print("\n评估结果（IoU匹配）：")
    print(results_iou.to_string(index=False))
    
    # 方法2：使用距离匹配（适用于点源检测）
    print("\n" + "=" * 80)
    print("方法2: 使用距离匹配")
    print("=" * 80)
    
    evaluator_dist = StarDetectionEvaluator()
    
    results_dist = evaluator_dist.evaluate_by_snr_ranges(
        gt_folder=gt_folder,
        pred_folder=pred_folder,
        snr_folder=snr_folder,
        snr_ranges=[(0, 5), (5, 10), (10, 20), (20, 50), (50, float('inf'))],
        use_distance_matching=True,
        distance_threshold=5.0  # 5像素距离阈值
    )
    
    print("\n评估结果（距离匹配）：")
    print(results_dist.to_string(index=False))
    
    # 方法3：使用更准确的全局评估
    print("\n" + "=" * 80)
    print("方法3: 全局评估（推荐）")
    print("=" * 80)
    
    results_global = evaluator_iou.evaluate_with_global_metrics(
        gt_folder=gt_folder,
        pred_folder=pred_folder,
        snr_folder=snr_folder
    )
    
    print("\n评估结果（全局匹配）：")
    print(results_global.to_string(index=False))
    
    # 可视化最佳结果
    evaluator_iou.plot_results(results_dist, save_path='evaluation_results.png')
    
    # 保存结果
    results_dist.to_csv('evaluation_results.csv', index=False)
    print(f"\n结果已保存到 evaluation_results.csv")
    
    # 打印详细分析
    print("\n" + "=" * 80)
    print("详细分析")
    print("=" * 80)
    
    for _, row in results_dist.iterrows():
        if row['snr_range'] != 'Overall':
            print(f"\nSNR范围: {row['snr_range']}")
            print(f"  - GT恒星数: {row['total_gt']}")
            print(f"  - 检测到的: {row['tp']} ({row['recall']*100:.1f}%)")
            print(f"  - 漏检的: {row['fn']}")


if __name__ == "__main__":
    main()

