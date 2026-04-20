# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams['axes.linewidth'] = 1.5

# color1 = '#F08F92'
# #F08F92'
# color2 = '#9CBEDB'
# #9CBEDB'
# color3 = '#A9D5A3'
# #A9D5A3'
# # 假设有三个数据集
# x_values = [0.4, 0.5, 0.6, 0.7, 0.8]
# x_new_values = [0, 5, 7, 10, 12]
# y1 = [23.31, 23.42, 23.65, 23.58, 23.48]
# y2 = [22.41, 22.21, 23.32, 23.29, 23.27]
# y3 = [23.70, 23.96, 23.79, 23.71, 23.58]

# # 计算柱状图宽度和x轴位置
# bar_width = 0.2
# x_ticks = np.arange(len(x_values))

# # 设置图形的大小
# plt.figure(figsize=(12, 4), dpi=1000)

# # 第一个子图 - 使用柱状图
# plt.subplot(1, 2, 1)
# bars1 = plt.bar(x_ticks - bar_width, y1, bar_width, label='Full', color=color1, edgecolor='gray', linewidth=1.5)
# bars2 = plt.bar(x_ticks, y2, bar_width, label='Rare', color=color2, edgecolor='gray', linewidth=1.5)
# bars3 = plt.bar(x_ticks + bar_width, y3, bar_width, label='Non-rare', color=color3, edgecolor='gray', linewidth=1.5)

# plt.ylim(21, 24.5)
# plt.yticks([21, 22, 23, 24], fontsize=14, fontweight='bold')  # 显示纵坐标刻度
# plt.xticks(x_ticks, x_values, fontsize=14, fontweight='bold')
# plt.legend(fontsize=9)

# # 添加纵轴标签 "mAP"
# plt.ylabel('mAP (%)', fontsize=16, fontweight='bold')

# # 添加文字标签 fig1
# plt.figtext(0.315, 0.035, 'Threshold', ha='center', fontsize=14, fontweight='bold')

# # 第二个子图 - 新的数据集
# y1_new = [20.64, 22.93, 23.65, 23.52, 23.20]
# y2_new = [20.51, 22.46, 23.32, 23.15, 23.04]
# y3_new = [21.12, 23.13, 23.79, 23.68, 23.56]

# # 计算新数据集的柱状图宽度和x轴位置
# x_ticks_new = np.arange(len(x_new_values))

# plt.subplot(1, 2, 2)
# bars1_new = plt.bar(x_ticks_new - bar_width, y1_new, bar_width, label='Full', color=color1, edgecolor='gray', linewidth=1.5)
# bars2_new = plt.bar(x_ticks_new, y2_new, bar_width, label='Rare', color=color2, edgecolor='gray', linewidth=1.5)
# bars3_new = plt.bar(x_ticks_new + bar_width, y3_new, bar_width, label='Non-rare', color=color3, edgecolor='gray', linewidth=1.5)

# plt.ylim(20, 24.5)
# plt.yticks([20, 21, 22, 23, 24], fontsize=14, fontweight='bold')  # 显示纵坐标刻度
# plt.xticks(x_ticks_new, x_new_values, fontsize=14, fontweight='bold')

# # 移动图例到右下角
# plt.legend(fontsize=9)

# # 添加纵轴标签 "mAP"
# plt.ylabel('mAP (%)', fontsize=16, fontweight='bold')

# # 添加文字标签 fig2
# plt.figtext(0.715, 0.035, 'Epoch', ha='center', fontsize=14, fontweight='bold')

# # 调整子图的大小，增加子图之间的间距
# plt.gcf().subplots_adjust(bottom=0.15, wspace=0.2)

# # 保存图形
# plt.savefig('bar_ablation.png')
# plt.savefig('ablation_2_bar.pdf', format='pdf')

# plt.show()









# ablation tu

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.linewidth'] = 1.5

color1 = '#A9D5A3'
color2 = '#9CBEDB'

# 原始数据
x_values = [0.4, 0.5, 0.6, 0.7, 0.8]
x_new_values = [0, 5, 7, 10, 12]
y1 = [23.31, 23.42, 23.65, 23.58, 23.48]  # Full
y2 = [22.41, 22.21, 23.32, 23.29, 23.27]  # Rare

y1_new = [20.64, 22.93, 23.65, 23.52, 23.20]
y2_new = [20.51, 22.46, 23.32, 23.15, 23.04]

bar_width = 0.25
x_ticks = np.arange(len(x_values))
x_ticks_new = np.arange(len(x_new_values))

plt.figure(figsize=(12, 4), dpi=1000)

# 第一个子图
ax1 = plt.subplot(1, 2, 1)
ax2 = ax1.twinx()



ax1.bar(x_ticks - bar_width/2, y1, bar_width, label='Full', color=color1, edgecolor='gray', linewidth=1.5)
ax2.bar(x_ticks + bar_width/2, y2, bar_width, label='Rare', color=color2, edgecolor='gray', linewidth=1.5)

ax1.set_ylim(21, 24.5)
ax1.set_yticks([21, 22, 23, 24])
ax1.tick_params(axis='y', labelsize=14, width=1.5)
ax1.set_yticklabels([21, 22, 23, 24], fontsize=14, fontweight='bold')

ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_values, fontsize=14,  fontweight='bold')

ax2.set_ylim(21, 24.5)
ax2.set_yticks([21, 22, 23, 24])
ax2.tick_params(axis='y', labelsize=14, width=1.5)
ax2.set_yticklabels([21, 22, 23, 24], fontsize=14, fontweight='bold')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=13)

plt.figtext(0.315, 0.035, r'$\tau_{low}$', ha='center', fontsize=26, fontweight='bold')

# 第二个子图
ax3 = plt.subplot(1, 2, 2)
ax4 = ax3.twinx()

ax3.bar(x_ticks_new - bar_width/2, y1_new, bar_width, label='Full', color=color1, edgecolor='gray', linewidth=1.5)
ax4.bar(x_ticks_new + bar_width/2, y2_new, bar_width, label='Rare', color=color2, edgecolor='gray', linewidth=1.5)

ax3.set_ylim(20, 24.5)
ax3.set_yticks([20, 21, 22, 23, 24])
ax3.tick_params(axis='y', labelsize=14, width=1.5)
ax3.set_yticklabels([20, 21, 22, 23, 24], fontsize=14, fontweight='bold')

ax3.set_xticks(x_ticks_new)
ax3.set_xticklabels(x_new_values, fontsize=14, fontweight='bold')

ax4.set_ylim(20, 24.5)
ax4.set_yticks([20, 21, 22, 23, 24])
ax4.tick_params(axis='y', labelsize=14, width=1.5)
ax4.set_yticklabels([20, 21, 22, 23, 24], fontsize=14, fontweight='bold')

lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, fontsize=13)


plt.figtext(0.715, 0.035, r'$\lambda$', ha='center', fontsize=26, fontweight='bold')

plt.gcf().subplots_adjust(bottom=0.15, wspace=0.2)
plt.savefig('data/tools/paper_view_tools/AAAI2026/ablation/bar_ablation_2bar.png')
plt.savefig('data/tools/paper_view_tools/AAAI2026/ablation/bar_ablation_2bar.pdf', format='pdf')
plt.show()

