
# import matplotlib.pyplot as plt
# import numpy as np
# import csv
# from matplotlib.lines import Line2D

# # 读取 CSV 文件数据
# def read_csv(file_path):
#     categories = []
#     values = []
#     with open(file_path, 'r', encoding='utf-8-sig') as file:  # 处理 BOM
#         reader = csv.reader(file, quotechar='"', skipinitialspace=True)
#         row_count = 0
#         for row in reader:
#             if row_count >= 11:  # 只读取前10行
#                 break
#             categories.append(row[0])  # 第一列为类名
#             values.append(list(map(int, row[1:4])))  # 只取前3列数值
#             row_count += 1
#     return categories, values

# # 加载数据
# csv_file = "figs/LAMOST/csv/img.csv"  # 替换为实际文件路径
# categories, values = read_csv(csv_file)

# # 设置空心圆半径
# inner_circle_radius = 1.6

# values = np.array(values, dtype=float)

# # 对数据差异大的数据取对数
# row_sums = values.sum(axis=1)
# log_sums = np.log2(row_sums)
# data = values / row_sums[:, None] * log_sums[:, None]




# # 设置类别数和角度
# num_categories = len(categories)  # 现在是10个
# angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

# colors = ['#F09FB6', '#90CDD1', '#A59CD4']

# # 标签（减少到3个）
# labels = ['Train', 'Dev', 'Test']

# # 创建极坐标图，调整尺寸以减少留白
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

# # 绘制每个柱子
# max_heights = []  # 用于存储每个柱子的总高度，便于设置标签位置
# bar_width = 1.4 * np.pi / num_categories  # 增加柱子宽度，减少间隙
# for i, angle in enumerate(angles):
#     bottom = inner_circle_radius 
#     for j in range(3):  # 从4改为3
#         height = data[i, j]
#         original_value = int(values[i, j])  # 获取原始数值用于显示
#         ax.bar(angle, height, width=bar_width, bottom=bottom, 
#                color=colors[j], edgecolor='white', linewidth=1.5, alpha=0.9)
        
#         # 在柱子中心添加数值标签
#         if height > 0.25:  # 降低显示文字的阈值
#             text_radius = bottom + height / 2  # 文字位置在柱子段的中心
#             text_angle = angle
            
#             # 根据角度调整文字方向
#             rotation = np.degrees(angle)
#             if 90 < rotation < 270:  # 左半圆，文字需要倒转
#                 text_rotation = rotation + 180
#                 ax.text(text_angle, text_radius, str(original_value), 
#                        rotation=text_rotation, fontsize=18, ha='center', va='center', 
#                        color='white', weight='bold', zorder=20)
#             else:  # 右半圆
#                 ax.text(text_angle, text_radius, str(original_value), 
#                        rotation=rotation, fontsize=18, ha='center', va='center', 
#                        color='white', weight='bold', zorder=20)
        
#         bottom += height  
#     max_heights.append(bottom)

# # 添加中心空心圆
# circle = plt.Circle((0, 0), inner_circle_radius, transform=ax.transData._b, 
#                     color='white', zorder=10)
# ax.add_artist(circle)

# # 添加图例 - 优化位置以减少留白
# legend_patches = [Line2D([0], [0], color=color, lw=8) for color in colors]

# # 将图例移到更合适的位置，减少留白
# ax.legend(
#     legend_patches,
#     labels,
#     loc='upper right',  
#     bbox_to_anchor=(1.05, 1.0), 
#     fontsize=13,
#     frameon=True,
#     fancybox=True,
#     shadow=True
# )

# # 设置类别标签位置和角度
# ax.set_xticks([])  
# ax.set_yticks([])

# for i, angle in enumerate(angles):
#     label = categories[i]  
#     max_height = max_heights[i]  # 获取当前柱子的最大高度
#     text_length = len(label)  # 计算文字长度
#     label_distance = max_height + 0.15 + text_length * 0.15   # 优化距离计算
#     rotation = np.degrees(angle)  # 标签的旋转角度
    
#     # 调整文字方向，确保可读性
#     if 90 < rotation < 270:  # 如果在左半圆
#         text_rotation = rotation + 180
#         ax.text(angle, label_distance, label, rotation=text_rotation,
#                 fontsize=16, ha='center', va='center', zorder=15, 
#                 weight='bold', color='#2C3E50')
#     else:  # 如果在右半圆
#         ax.text(angle, label_distance, label, rotation=rotation,
#                 fontsize=16, ha='center', va='center', zorder=15,
#                 weight='bold', color='#2C3E50')

# # 隐藏径向网格线和坐标轴
# ax.yaxis.grid(False)
# ax.spines['polar'].set_visible(False)



# # 设置图表背景
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')

# # 保存高质量PNG - 优化布局设置
# plt.tight_layout()
# plt.savefig('figs/LAMOST/dataset/imgNumber.png', 
#             dpi=500,  # 调整DPI以平衡质量和文件大小
#             bbox_inches='tight',  # 移除多余的白边
#             pad_inches=0.05,  # 减少边距
#             facecolor='white',
#             edgecolor='none')

# # 保存高质量PDF
# plt.savefig('figs/LAMOST/dataset/imgNumber.pdf', 
#             format='pdf',
#             dpi=300,
#             bbox_inches='tight',
#             pad_inches=0.05,
#             facecolor='white',
#             edgecolor='none')







import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.lines import Line2D

# 读取 CSV 文件数据
def read_csv(file_path):
    categories = []
    values = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:  # 处理 BOM
        reader = csv.reader(file, quotechar='"', skipinitialspace=True)
        row_count = 0
        for row in reader:
            if row_count >= 11:  # 只读取前10行
                break
            categories.append(row[0])  # 第一列为类名
            values.append(list(map(int, row[1:4])))  # 只取前3列数值
            row_count += 1
    return categories, values

# 加载数据
csv_file = "figs/LAMOST/csv/img.csv"  # 替换为实际文件路径
categories, values = read_csv(csv_file)

# 设置空心圆半径
inner_circle_radius = 1.6

values = np.array(values, dtype=float)

# 对数据差异大的数据取对数
row_sums = values.sum(axis=1)
log_sums = np.log2(row_sums)
data = values / row_sums[:, None] * log_sums[:, None]




# 设置类别数和角度
num_categories = len(categories)  # 现在是10个
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

colors = ['#F09FB6', '#90CDD1', '#A59CD4']

# 标签（减少到3个）
labels = ['Train', 'Dev', 'Test']

# 创建极坐标图，调整尺寸以减少留白
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

# 绘制每个柱子
max_heights = []  # 用于存储每个柱子的总高度，便于设置标签位置
bar_width = 1.4 * np.pi / num_categories  # 增加柱子宽度，减少间隙
for i, angle in enumerate(angles):
    bottom = inner_circle_radius 
    for j in range(3):  # 从4改为3
        height = data[i, j]
        original_value = int(values[i, j])  # 获取原始数值用于显示
        ax.bar(angle, height, width=bar_width, bottom=bottom, 
               color=colors[j], edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # 在柱子中心添加数值标签
        if height > 0.25:  # 降低显示文字的阈值
            text_radius = bottom + height / 2  # 文字位置在柱子段的中心
            text_angle = angle
            
            # 根据角度调整文字方向
            rotation = np.degrees(angle)
            if 90 < rotation < 270:  # 左半圆，文字需要倒转
                text_rotation = rotation + 180
                ax.text(text_angle, text_radius, str(original_value), 
                       rotation=text_rotation, fontsize=21, ha='center', va='center', 
                       color='white', weight='bold', zorder=20)
            else:  # 右半圆
                ax.text(text_angle, text_radius, str(original_value), 
                       rotation=rotation, fontsize=21, ha='center', va='center', 
                       color='white', weight='bold', zorder=20)
        
        bottom += height  
    max_heights.append(bottom)

# 添加中心空心圆
circle = plt.Circle((0, 0), inner_circle_radius, transform=ax.transData._b, 
                    color='white', zorder=10)
ax.add_artist(circle)

# # 添加图例 - 优化位置以减少留白
# legend_patches = [Line2D([0], [0], color=color, lw=8) for color in colors]

# # 将图例移到更合适的位置，减少留白
# ax.legend(
#     legend_patches,
#     labels,
#     loc='upper right',  
#     bbox_to_anchor=(1.05, 1.0), 
#     fontsize=20,
#     frameon=True,
#     fancybox=True,
#     shadow=True
# )


# 设置类别标签位置和角度 - 修改为与柱子垂直
ax.set_xticks([])  
ax.set_yticks([])

for i, angle in enumerate(angles):
    label = categories[i]  
    max_height = max_heights[i]  # 获取当前柱子的最大高度
    text_length = len(label)  # 计算文字长度
    label_distance = max_height + 0.15 + text_length * 0.15   # 优化距离计算
    
    # 修改：让标签与柱子垂直（切线方向）
    rotation = np.degrees(angle) + 90  # 与径向垂直就是加90度
    
    # 为了保持文字可读性，避免文字倒置
    if rotation > 90 and rotation < 270:
        rotation += 180  # 如果文字会倒置，就再旋转180度
    
    ax.text(angle, label_distance, label, rotation=rotation,
            fontsize=24, ha='center', va='center', zorder=15, 
            weight='bold', color='#2C3E50')

# 隐藏径向网格线和坐标轴
ax.yaxis.grid(False)
ax.spines['polar'].set_visible(False)



# 设置图表背景
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 保存高质量PNG - 优化布局设置
plt.tight_layout()
plt.savefig('figs/LAMOST/dataset/imgNumber.png', 
            dpi=800,  # 调整DPI以平衡质量和文件大小
            bbox_inches='tight',  # 移除多余的白边
            pad_inches=0.05,  # 减少边距
            facecolor='white',
            edgecolor='none')

# 保存高质量PDF
plt.savefig('figs/LAMOST/dataset/imgNumber.pdf', 
            format='pdf',
            dpi=500,
            bbox_inches='tight',
            pad_inches=0.05,
            facecolor='white',
            edgecolor='none')






# # 水滴状柱状图
# import matplotlib.pyplot as plt
# import numpy as np
# import csv
# from matplotlib.lines import Line2D

# # 读取 CSV 文件数据
# def read_csv(file_path):
#     categories = []
#     values = []
#     with open(file_path, 'r', encoding='utf-8-sig') as file:  # 处理 BOM
#         reader = csv.reader(file, quotechar='"', skipinitialspace=True)
#         row_count = 0
#         for row in reader:
#             if row_count >= 11:  # 只读取前10行
#                 break
#             categories.append(row[0])  # 第一列为类名
#             values.append(list(map(int, row[1:4])))  # 只取前3列数值
#             row_count += 1
#     return categories, values

# # 加载数据
# csv_file = "figs/LAMOST/csv/img.csv"  # 替换为实际文件路径
# categories, values = read_csv(csv_file)

# # 设置空心圆半径
# inner_circle_radius = 1.6

# values = np.array(values, dtype=float)

# # 对数据差异大的数据取对数
# row_sums = values.sum(axis=1)
# log_sums = np.log2(row_sums)
# data = values / row_sums[:, None] * log_sums[:, None]

# # 设置类别数和角度
# num_categories = len(categories)  # 现在是10个
# angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

# colors = ['#FF6B9D', '#4ECDC4', '#A78BFA']

# # 标签（减少到3个）
# labels = ['Train', 'Dev', 'Test']

# # 创建极坐标图，调整尺寸以减少留白
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

# # 优化的水滴形状函数
# def create_petal_shape(center_angle, inner_radius, outer_radius, width_factor=0.6, layer_index=0):
#     """创建优化的水滴/花瓣形状的坐标点"""
#     # 生成角度范围
#     angle_range = width_factor * np.pi / num_categories
#     theta_points = np.linspace(center_angle - angle_range/2, center_angle + angle_range/2, 50)
    
#     # 创建更平滑的水滴形状
#     normalized_theta = np.linspace(0, np.pi, len(theta_points))
    
#     # 根据层级调整形状参数，让内层（Train）不那么尖锐
#     if layer_index == 0:  # Train层，使用更平缓的曲线
#         shape_power = 1.2  # 更大的指数让形状更平缓
#         base_factor = 0.3  # 底部更宽
#     elif layer_index == 1:  # Dev层
#         shape_power = 1.0
#         base_factor = 0.2
#     else:  # Test层
#         shape_power = 0.8
#         base_factor = 0.1
    
#     # 改进的径向距离计算
#     # 使用组合函数创建更自然的水滴形状
#     sin_profile = np.sin(normalized_theta) ** shape_power
#     # 添加基础宽度，避免过于尖锐
#     base_width = base_factor * np.sin(normalized_theta)
#     radius_profile = np.maximum(sin_profile, base_width)
    
#     # 平滑过渡
#     radius_profile = np.clip(radius_profile, 0.1, 1.0)  # 确保最小宽度
    
#     # 计算实际的径向距离
#     radius_points = inner_radius + (outer_radius - inner_radius) * radius_profile
    
#     return theta_points, radius_points

# # 绘制每个水滴形状
# max_heights = []  # 用于存储每个水滴的最大高度，便于设置标签位置

# for i, angle in enumerate(angles):
#     bottom = inner_circle_radius 
    
#     for j in range(3):  # 3个数据系列
#         height = data[i, j]
#         original_value = int(values[i, j])  # 获取原始数值用于显示
        
#         if height > 0:  # 只绘制有数据的部分
#             # 创建优化的水滴形状，传入层级信息
#             theta_points, radius_points = create_petal_shape(
#                 angle, bottom, bottom + height, width_factor=0.8, layer_index=j
#             )
            
#             # 绘制填充的水滴形状
#             ax.fill(theta_points, radius_points, color=colors[j], 
#                    alpha=0.9, edgecolor='white', linewidth=1.5, zorder=10)
            
#             # 在水滴中心添加数值标签
#             if height > 0.25:  # 降低显示文字的阈值
#                 text_radius = bottom + height * 0.6  # 文字位置在水滴的较上位置
#                 text_angle = angle
                
#                 # 根据角度调整文字方向
#                 rotation = np.degrees(angle)
#                 if 90 < rotation < 270:  # 左半圆，文字需要倒转
#                     text_rotation = rotation + 180
#                     ax.text(text_angle, text_radius, str(original_value), 
#                            rotation=text_rotation, fontsize=11, ha='center', va='center', 
#                            color='white', weight='bold', zorder=20)
#                 else:  # 右半圆
#                     ax.text(text_angle, text_radius, str(original_value), 
#                            rotation=rotation, fontsize=11, ha='center', va='center', 
#                            color='white', weight='bold', zorder=20)
        
#         bottom += height  
#     max_heights.append(bottom)

# # 添加中心空心圆 - 使用更大的圆来适配水滴形状
# circle = plt.Circle((0, 0), inner_circle_radius, transform=ax.transData._b, 
#                     color='white', zorder=15, edgecolor='#E0E0E0', linewidth=2)
# ax.add_artist(circle)

# # 添加图例 - 优化位置以减少留白
# legend_patches = [Line2D([0], [0], color=color, lw=8) for color in colors]

# # 将图例移到更合适的位置，减少留白
# ax.legend(
#     legend_patches,
#     labels,
#     loc='upper right',  
#     bbox_to_anchor=(1.05, 1.0), 
#     fontsize=12,
#     frameon=True,
#     fancybox=True,
#     shadow=True
# )

# # 设置类别标签位置和角度
# ax.set_xticks([])  
# ax.set_yticks([])

# for i, angle in enumerate(angles):
#     label = categories[i]  
#     max_height = max_heights[i]  # 获取当前柱子的最大高度
#     text_length = len(label)  # 计算文字长度
#     label_distance = max_height + 0.15 + text_length * 0.15   # 优化距离计算
#     rotation = np.degrees(angle)  # 标签的旋转角度
    
#     # 调整文字方向，确保可读性
#     if 90 < rotation < 270:  # 如果在左半圆
#         text_rotation = rotation + 180
#         ax.text(angle, label_distance, label, rotation=text_rotation,
#                 fontsize=14, ha='center', va='center', zorder=15, 
#                 weight='bold', color='#2C3E50')
#     else:  # 如果在右半圆
#         ax.text(angle, label_distance, label, rotation=rotation,
#                 fontsize=14, ha='center', va='center', zorder=15,
#                 weight='bold', color='#2C3E50')

# # 隐藏径向网格线和坐标轴
# ax.yaxis.grid(False)
# ax.spines['polar'].set_visible(False)

# # 设置图表背景
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')

# # 保存高质量PNG - 优化布局设置
# plt.tight_layout()
# plt.savefig('figs/LAMOST/dataset/imgNumber.png', 
#             dpi=300,  # 调整DPI以平衡质量和文件大小
#             bbox_inches='tight',  # 移除多余的白边
#             pad_inches=0.05,  # 减少边距
#             facecolor='white',
#             edgecolor='none')

# # 保存高质量PDF
# plt.savefig('figs/LAMOST/dataset/imgNumber.pdf', 
#             format='pdf',
#             dpi=300,
#             bbox_inches='tight',
#             pad_inches=0.05,
#             facecolor='white',
#             edgecolor='none')