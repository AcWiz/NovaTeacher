# 原始郭哥版

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.linewidth'] = 1.5

color1 = '#F08F92'
color2 = '#9CBEDB'
color3 = '#A9D5A3'
# 假设有三个数据集
x_values = [ 0.4, 0.6, 0.8, 1.0]
x_new_values = [0.3, 0.5, 0.7, 0.9]
y1 = [23.31, 23.42, 23.65, 23.58, 23.48]
y2 = [22.41, 22.21, 23.32, 23.29, 23.27]
y3 = [23.70, 23.96, 23.79, 23.71, 23.58]

# 计算柱状图宽度和x轴位置
bar_width = 0.2
x_ticks = np.arange(len(x_values))


# 设置图形的大小
plt.figure(figsize=(12, 4), dpi=1000)

# 第一个子图 - 使用柱状图
plt.subplot(1, 2, 1)
bars1 = plt.bar(x_ticks - bar_width, y1, bar_width, label='Full', color=color1, edgecolor='gray', linewidth=1.5)
bars2 = plt.bar(x_ticks, y2, bar_width, label='Rare', color=color2, edgecolor='gray', linewidth=1.5)
bars3 = plt.bar(x_ticks + bar_width, y3, bar_width, label='Non-rare', color=color3, edgecolor='gray', linewidth=1.5)


plt.ylim(21, 24.5)
plt.yticks([21, 22, 23, 24], fontsize=9, fontweight='bold')  # 显示纵坐标刻度
plt.xticks(x_ticks, x_values, fontsize=9, fontweight='bold')
plt.legend(fontsize=9)

# 添加纵轴标签 "mAP"
plt.ylabel('mAP (%)', fontsize=12, fontweight='bold')

# 添加文字标签 fig1
plt.figtext(0.315, 0.035, 'Threshold', ha='center', fontsize=14, fontweight='bold')

# 第二个子图 - 新的数据集
y1_new = [20.64, 22.93, 23.65, 23.52, 23.20]
y2_new = [20.51, 22.46, 23.32, 23.15, 23.04]
y3_new = [21.12, 23.13, 23.79, 23.68, 23.56]

# 计算新数据集的柱状图宽度和x轴位置
x_ticks_new = np.arange(len(x_new_values))

plt.subplot(1, 2, 2)
bars1_new = plt.bar(x_ticks_new - bar_width, y1_new, bar_width, label='Full', color=color1, edgecolor='gray', linewidth=1.5)
bars2_new = plt.bar(x_ticks_new, y2_new, bar_width, label='Rare', color=color2, edgecolor='gray', linewidth=1.5)
bars3_new = plt.bar(x_ticks_new + bar_width, y3_new, bar_width, label='Non-rare', color=color3, edgecolor='gray', linewidth=1.5)



plt.ylim(20, 24.5)
plt.yticks([20, 21, 22, 23, 24], fontsize=9, fontweight='bold')  # 显示纵坐标刻度
plt.xticks(x_ticks_new, x_new_values, fontsize=9, fontweight='bold')

# 移动图例到右下角
plt.legend(fontsize=9)

# 添加纵轴标签 "mAP"
plt.ylabel('mAP (%)', fontsize=12, fontweight='bold')

# 添加文字标签 fig2
plt.figtext(0.715, 0.035, 'Epoch', ha='center', fontsize=14, fontweight='bold')

# 调整子图的大小，增加子图之间的间距
plt.gcf().subplots_adjust(bottom=0.15, wspace=0.2)

# 保存图形
plt.savefig('data/tools/paper_view_tools/ablation/bar_ablation.png')
plt.savefig('data/tools/paper_view_tools/ablation/ablation_2_bar.pdf', format='pdf')

plt.show()





