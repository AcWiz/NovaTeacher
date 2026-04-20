






# import matplotlib.pyplot as plt
# import numpy as np

# # 设置样式 - 不使用网格样式
# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['axes.grid'] = False  # 确保不显示网格

# color1 = '#F09FB6'
# color2 = '#90CDD1'

# # 数据
# x_values = ["original", "potential"]
# y1 = [37633, 117871]  # 第一组数据
# y2 = [21859, 51615]    # 第二组数据

# # 计算柱状图宽度和x轴位置
# bar_width = 0.25  # 增加间隙，使柱子更加细长
# x_ticks = np.arange(len(x_values)) * 0.75  # 缩小簇之间的距离

# # 设置图形的大小
# fig, ax = plt.subplots(figsize=(8, 6), dpi=500)

# # 绘制柱状图 - 添加透明度和渐变效果
# bars1 = ax.bar(x_ticks - bar_width / 2, y1, bar_width, 
#                label='Train', color=color1, alpha=0.9,
#                edgecolor='white', linewidth=2.5,
#                capsize=5)

# bars2 = ax.bar(x_ticks + bar_width / 2, y2, bar_width, 
#                label='Val', color=color2, alpha=0.9,
#                edgecolor='white', linewidth=2.5,
#                capsize=5)

# # 添加柱子中间的数值标签（竖排）- 增大字体，显示完整数字
# def add_value_labels(bars, values):
#     for bar, value in zip(bars, values):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height/2,
#                 f'{value:,}',  # 显示完整数字，添加千位分隔符
#                 ha='center', va='center', fontsize=18, fontweight='bold',
#                 color='white', rotation=90)

# add_value_labels(bars1, y1)
# add_value_labels(bars2, y2)

# # 设置坐标轴样式 - 放大字号并加粗
# ax.tick_params(axis='y', labelsize=22, colors='#070707')
# ax.tick_params(axis='x', labelsize=22, colors="#070707")

# # 去掉y轴的刻度标签和刻度线（不显示纵坐标数值）
# ax.set_yticklabels([])
# ax.yaxis.set_ticks_position('none')

# ax.set_xticks(x_ticks)
# ax.set_xticklabels(x_values, fontweight='bold')

# # 创建更大的图例 - 增大字体
# legend = ax.legend(fontsize=20, frameon=True, fancybox=True, 
#                    shadow=True, framealpha=0.9,
#                    loc='upper left', bbox_to_anchor=(0.02, 0.98))
# legend.get_frame().set_edgecolor('#BDC3C7')
# legend.get_frame().set_linewidth(1.5)

# # 设置干净的背景
# ax.set_facecolor('white')
# fig.patch.set_facecolor('white')

# # 设置图形边界框
# fig.patch.set_edgecolor('black')
# fig.patch.set_linewidth(2)

# # 移除顶部和右侧的边框
# ax.spines['top'].set_visible(True)
# ax.spines['right'].set_visible(True)
# ax.spines['left'].set_color("#0C0C0C")
# ax.spines['bottom'].set_color('#0C0C0C')

# # 调整布局
# plt.tight_layout()

# # 保存图形
# plt.savefig('data/tools/paper_view_tools/AAAI2026/dataset_guo/bar_ablation_enhanced.png', 
#             dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')
# plt.savefig('data/tools/paper_view_tools/AAAI2026/dataset_guo/ablation_2_bar_enhanced.pdf', 
#             format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')

# plt.show()





import matplotlib.pyplot as plt
import numpy as np

# 设置样式 - 不使用网格样式
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid'] = False  # 确保不显示网格

color1 = '#A9D5A3'
color2 = '#9CBEDB'

# 数据 1 
x_values = ["Images", "Instances"]
y1 = [37633, 117871]  # 第一组数据
y2 = [21859, 51615]    # 第二组数据


# # 数据 2 
# x_values = ["Images", "Instances"]
# y1 = [5400, 10364]  # 第一组数据
# y2 = [4965, 8863]    # 第二组数据

# 计算柱状图宽度和x轴位置
bar_width = 0.3  # 增加间隙，使柱子更加细长
x_ticks = np.arange(len(x_values)) * 0.85  # 缩小簇之间的距离，增大两簇之间的间距

# 设置图形的大小
fig, ax = plt.subplots(figsize=(8, 6), dpi=500)

# 绘制柱状图 - 添加透明度和渐变效果
bars1 = ax.bar(x_ticks - bar_width / 2, y1, bar_width, 
               label='Labeled', color=color1, alpha=0.9,
               edgecolor='white', linewidth=2.5,
               capsize=5)

bars2 = ax.bar(x_ticks + bar_width / 2, y2, bar_width, 
               label='Unlabeled', color=color2, alpha=0.9,
               edgecolor='white', linewidth=2.5,
               capsize=5)



# 设置坐标轴样式 - 放大字号并加粗
ax.tick_params(axis='y', labelsize=22, colors='#070707')
ax.tick_params(axis='x', labelsize=22, colors="#070707")

# 去掉y轴的刻度标签和刻度线（不显示纵坐标数值）
ax.set_yticklabels([])
ax.yaxis.set_ticks_position('none')

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_values, fontweight='bold')

# 创建更大的图例 - 增大字体
legend = ax.legend(fontsize=24, frameon=True, fancybox=True, 
                   shadow=True, framealpha=0.9,
                   loc='upper left', bbox_to_anchor=(0.02, 0.98))
legend.get_frame().set_edgecolor('#BDC3C7')
legend.get_frame().set_linewidth(1.5)

# 设置干净的背景
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 设置图形边界框
fig.patch.set_edgecolor('black')
fig.patch.set_linewidth(2)

# 移除顶部和右侧的边框
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_color("#0C0C0C")
ax.spines['bottom'].set_color('#0C0C0C')

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig('data/tools/paper_view_tools/AAAI2026/dataset_guo/dataset_bar_1.png', 
            dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('data/tools/paper_view_tools/AAAI2026/dataset_guo/dataset_bar_1.pdf', 
            format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()

