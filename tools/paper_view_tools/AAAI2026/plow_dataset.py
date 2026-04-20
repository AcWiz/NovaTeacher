
# # 数据集分布柱状图 - 美化版本

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle
# import matplotlib.patches as mpatches

# # 设置样式 - 不使用网格样式
# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['axes.grid'] = False  # 确保不显示网格

# # # 优化的配色方案 - 更加和谐的颜色
# # color1 = '#FF6B9D'  # 粉色系
# # color2 = '#4ECDC4'  # 青色系  
# # color3 = '#A8E6CF'  # 绿色系

# color1 = '#F09FB6'
# color2 = '#90CDD1'
# color3 = '#A59CD4'



# # 数据
# x_values = ["Extreme", "Intermediate", "Trivial"]
# y1 = [243573, 116338, 80217]
# y2 = [79803, 38405, 26398]
# y3 = [80681, 37892, 25591]

# # 计算柱状图宽度和x轴位置
# bar_width = 0.25
# x_ticks = np.arange(len(x_values))

# # 设置图形的大小
# fig, ax = plt.subplots(figsize=(12, 12), dpi=500)

# # 绘制柱状图 - 添加透明度和渐变效果
# bars1 = ax.bar(x_ticks - bar_width, y1, bar_width, 
#                label='Train', color=color1, alpha=0.9,
#                edgecolor='white', linewidth=2.5,
#                capsize=5)

# bars2 = ax.bar(x_ticks, y2, bar_width, 
#                label='Dev', color=color2, alpha=0.9,
#                edgecolor='white', linewidth=2.5,
#                capsize=5)

# bars3 = ax.bar(x_ticks + bar_width, y3, bar_width, 
#                label='Test', color=color3, alpha=0.9,
#                edgecolor='white', linewidth=2.5,
#                capsize=5)

# # 添加柱子中间的数值标签（竖排）
# def add_value_labels(bars, values):
#     for bar, value in zip(bars, values):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height/2,
#                 f'{value:,}',  # 添加千位分隔符
#                 ha='center', va='center', fontsize=17, fontweight='bold',
#                 color='white', rotation=90)

# add_value_labels(bars1, y1)
# add_value_labels(bars2, y2)
# add_value_labels(bars3, y3)

# # 设置坐标轴样式 - 放大字号并加粗
# ax.tick_params(axis='y', labelsize=28, colors='#070707')
# ax.tick_params(axis='x', labelsize=30, colors="#070707")

# # 设置Y轴标签为粗体，并添加千位分隔符
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
# for label in ax.get_yticklabels():
#     label.set_fontweight('bold')

# ax.set_xticks(x_ticks)
# ax.set_xticklabels(x_values, fontweight='bold')

# # # 创建更美观的图例
# legend = ax.legend(fontsize=25, frameon=True, fancybox=True, 
#                    shadow=True, framealpha=0.9,
#                    loc='upper right', bbox_to_anchor=(0.98, 0.98))
# # legend.get_frame().set_facecolor('#F8F9FA')
# legend.get_frame().set_edgecolor('#BDC3C7')
# legend.get_frame().set_linewidth(1.5)

# # 设置干净的背景
# ax.set_facecolor('white')
# fig.patch.set_facecolor('white')

# # 添加标题（可选）
# # ax.set_title('Dataset Distribution', fontsize=18, fontweight='bold', 
# #              color='#2C3E50', pad=20)

# # 设置Y轴范围，留出更多空间给标签
# max_value = max(max(y1), max(y2), max(y3))
# ax.set_ylim(0, max_value * 1.15)

# # 移除顶部和右侧的边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_color("#0C0C0C")
# ax.spines['bottom'].set_color('#0C0C0C')

# # 调整布局
# plt.tight_layout()

# # 保存图形
# plt.savefig('data/tools/paper_view_tools/AAAI2026/dateset/bar_ablation_enhanced.png', 
#             dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')
# plt.savefig('data/tools/paper_view_tools/AAAI2026/dateset/ablation_2_bar_enhanced.pdf', 
#             format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')

# plt.show()

# # 如果你想要更加简洁的版本（没有数值标签），可以使用下面的代码：
# """
# # 简洁版本 - 不显示数值标签
# fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=300)

# bars1 = ax2.bar(x_ticks - bar_width, y1, bar_width, 
#                 label='Train', color=color1, alpha=0.8,
#                 edgecolor='white', linewidth=2)

# bars2 = ax2.bar(x_ticks, y2, bar_width, 
#                 label='Dev', color=color2, alpha=0.8,
#                 edgecolor='white', linewidth=2)

# bars3 = ax2.bar(x_ticks + bar_width, y3, bar_width, 
#                 label='Test', color=color3, alpha=0.8,
#                 edgecolor='white', linewidth=2)

# ax2.set_xticks(x_ticks)
# ax2.set_xticklabels(x_values, fontsize=14, fontweight='bold')
# ax2.tick_params(axis='y', labelsize=12)
# ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
# ax2.grid(True, alpha=0.3)
# ax2.set_facecolor('#FAFBFC')

# plt.tight_layout()
# plt.show()
# """






# 数据集分布柱状图 - 优化版本

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置样式 - 不使用网格样式
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid'] = False  # 确保不显示网格

color1 = '#F09FB6'
color2 = '#90CDD1'
color3 = '#A59CD4'

# 数据
x_values = ["Extreme", "Intermediate", "Trivial"]
y1 = [243573, 116338, 80217]
y2 = [79803, 38405, 26398]
y3 = [80681, 37892, 25591]

# 计算柱状图宽度和x轴位置
bar_width = 0.25
x_ticks = np.arange(len(x_values))

# 设置图形的大小
fig, ax = plt.subplots(figsize=(12, 15), dpi=500)

# 绘制柱状图 - 添加透明度和渐变效果
bars1 = ax.bar(x_ticks - bar_width, y1, bar_width, 
               label='Train', color=color1, alpha=0.9,
               edgecolor='white', linewidth=2.5,
               capsize=5)

bars2 = ax.bar(x_ticks, y2, bar_width, 
               label='Val', color=color2, alpha=0.9,
               edgecolor='white', linewidth=2.5,
               capsize=5)

bars3 = ax.bar(x_ticks + bar_width, y3, bar_width, 
               label='Test', color=color3, alpha=0.9,
               edgecolor='white', linewidth=2.5,
               capsize=5)

# 添加柱子中间的数值标签（竖排）- 增大字体，显示完整数字
def add_value_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{value:,}',  # 显示完整数字，添加千位分隔符
                ha='center', va='center', fontsize=26, 
                fontweight='bold',  # 从17增加到22
                color='white', rotation=90)

add_value_labels(bars1, y1)
add_value_labels(bars2, y2)
add_value_labels(bars3, y3)

# 自定义Y轴格式化函数
def format_y_axis(value, tick_number):
    """格式化Y轴刻度标签为k格式"""
    if value >= 1000000:
        return f'{value/1000000:.1f}M'
    elif value >= 1000:
        return f'{value/1000:.0f}k'
    else:
        return f'{int(value)}'

# 设置坐标轴样式 - 放大字号并加粗
ax.tick_params(axis='y', labelsize=28, colors='#070707')
ax.tick_params(axis='x', labelsize=30, colors="#070707")

# 设置Y轴标签格式化为k格式
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis))
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_values, fontweight='bold')

# 创建更大的图例 - 增大字体
legend = ax.legend(fontsize=48, frameon=True, fancybox=True,  # 从25增加到32
                   shadow=True, framealpha=0.9,
                   loc='upper right', bbox_to_anchor=(0.98, 0.98))
legend.get_frame().set_edgecolor('#BDC3C7')
legend.get_frame().set_linewidth(1.5)

# 设置干净的背景
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 设置Y轴范围，最大值设置为250k
ax.set_ylim(0, 250000)  # 250k = 250000

# 移除顶部和右侧的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color("#0C0C0C")
ax.spines['bottom'].set_color('#0C0C0C')

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig('data/tools/paper_view_tools/AAAI2026/dateset/bar_ablation_optimized.png', 
            dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('data/tools/paper_view_tools/AAAI2026/dateset/ablation_2_bar_optimized.pdf', 
            format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()







