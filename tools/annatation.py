
import pandas as pd
import numpy as np


def xywhaTodota(rbboxes):
    rbboxes.astype(float)
    x = rbboxes[:, 0].reshape(-1, 1).astype(float)
    y = rbboxes[:, 1].reshape(-1, 1).astype(float)
    x = np.round(x).astype(float)
    y = np.round(y).astype(float)
    # x = np.round(x).astype(int)
    # y = np.round(y).astype(int)
    w = rbboxes[:, 2].reshape(-1, 1).astype(int)
    h = rbboxes[:, 3].reshape(-1, 1).astype(int)
    a = rbboxes[:, 4].reshape(-1, 1).astype(float)


    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.concatenate([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], axis=-1)

    return polys



# 指定需要读取的列，并指定它们的dtype以防止混合类型警告
column_indices = [6, 9]  # 第三和第四列
dtype_spec = {col: 'str' for col in column_indices}

# 读取文件，只读取需要的列
df = pd.read_csv('data_denseStar/log/output_file.csv', 
                 delim_whitespace=True, 
                 header=None, 
                 usecols=column_indices, 
                 dtype=dtype_spec,
                 low_memory=False)

print(df)
# df = df.round().astype(int)
# print(df)

df['New_Column_1'] = 1.824
df['New_Column_2'] = 1.824
df['New_Column_3'] = 0.0
df['New_Column_4'] = 'star'

np_df = df.to_numpy()

dota_txt = xywhaTodota(np_df[1:,:5])



np.savetxt('data/dota_annotation.txt', dota_txt, fmt='%d') 

with open('data/06_img_annotation_wh4.txt','w+') as ft:
    for d_txt in dota_txt:

        x1,y1,x2,y2,x3,y3,x4,y4 = d_txt
        ft.write("%d " % x1)
        ft.write("%d " % y1)
        ft.write("%d " % x2)
        ft.write("%d " % y2)
        ft.write("%d " % x3)
        ft.write("%d " % y3)
        ft.write("%d " % x4)
        ft.write("%d " % y4)
        ft.write("star\n")
        # print(type(x1))

# df.to_csv('data/annotation.txt', index=False, header=False, sep='\t')