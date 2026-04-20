
import pandas as pd
import numpy as np
import os


def xywhaTodota(rbboxes):
    rbboxes[:,:5].astype(float)
    x = rbboxes[:, 0].reshape(-1, 1).astype(float)
    y = rbboxes[:, 1].reshape(-1, 1).astype(float)
    x = np.round(x).astype(float)
    y = np.round(y).astype(float)
    # x = np.round(x).astype(int)
    # y = np.round(y).astype(int)
    w = rbboxes[:, 2].reshape(-1, 1).astype(float) * 2
    h = rbboxes[:, 3].reshape(-1, 1).astype(float) * 2 
    a = rbboxes[:, 4].reshape(-1, 1).astype(float)
    label = rbboxes[:,5].reshape(-1, 1)


    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.concatenate([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, label], axis=-1)
    return polys



column_indices = [3, 4, 11]  # 第三、第四和第十一列

# 指定列的 dtype
dtype_spec = {3: 'str', 4: 'str', 11: 'str'}

root_dir = 'data/galaxy/galaxy_data_moved_ori/cat'
file_list = os.listdir(root_dir)

for file_ in file_list:
    # 读取文件，只读取需要的列，并指定列的 dtype
    df = pd.read_csv(os.path.join(root_dir, file_), 
                    delim_whitespace=True, 
                    header=None, 
                    usecols=column_indices, 
                    dtype=dtype_spec,
                    low_memory=False)


    # 添加新列，并根据条件赋值
    # df['New_Column_1'] = np.where(df[11] == 'star', 1.824, 3)
    # df['New_Column_2'] = np.where(df[11] == 'star', 1.824, 3)
    df['New_Column_1'] = 1.824
    df['New_Column_2'] = 1.824
    df['New_Column_3'] = 0.0
    
    cols = df.columns.tolist()
    cols.append(cols.pop(cols.index(11)))
    df = df[cols]


    cols = df.columns.tolist()
    cols.append(cols.pop(cols.index(11)))
    df = df[cols]


    # print(df)
    np_df = df.to_numpy()

    dota_txt = xywhaTodota(np_df[1:,:])
    txt_file = 'CSST_' + file_[-18:-14] + '_V01.txt'




    with open(os.path.join('data/galaxy/galaxy_data_moved_ori/annotation_allg3', txt_file),'w+') as ft:
        for d_txt in dota_txt:
            # for value in dota_txt:
            #     ft.write(f"{value:.0f} ")
            # print(str(d_txt) + 'star')
            x1,y1,x2,y2,x3,y3,x4,y4,label = d_txt
            ft.write("%f " % x1)
            ft.write("%f " % y1)
            ft.write("%f " % x2)
            ft.write("%f " % y2)
            ft.write("%f " % x3)
            ft.write("%f " % y3)
            ft.write("%f " % x4)
            ft.write("%f " % y4)
            ft.write("%s\n" % label)
            print(type(x1))

# 保存处理后的文件
# df.to_csv('data/annotation.txt', index=False, header=False, sep='\t')
