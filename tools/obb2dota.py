import numpy as np
import os
import pandas as pd


def xywhaTodota(rbboxes):
    x, y, w, h, a = rbboxes
    # a = np.radians(a)
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w * cosa, w * sina
    hx, hy = -h * sina, h * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y


input_dir = '/home/fenglonghan/projects/mmrotate/data/CSST_data/gt_converted'
out_dir = '/home/fenglonghan/projects/mmrotate/data/CSST_data/annos'
os.makedirs(out_dir, exist_ok=True)

# # Read Excel file (the .csv is actually an xlsx file)
# df = pd.read_excel(input_path)

# # Map columns: A_IMAGE and B_IMAGE are semi-axes, multiply by 2 for full width/height
# x = df['XWIN_IMAGE'].values
# y = df['YWIN_IMAGE'].values
# w = df['A_IMAGE'].values * 2
# h = df['B_IMAGE'].values * 2
# a = df['THETA_IMAGE'].values

# output_path = os.path.join(out_dir, 'image_01.txt')
# with open(output_path, 'w') as outfile:
#     for i in range(len(x)):
#         obb = np.array([x[i], y[i], w[i], h[i], a[i]], dtype=float)
#         label = 'star'
#         p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = xywhaTodota(obb)
#         outfile.write("%f " % p1x)
#         outfile.write("%f " % p1y)
#         outfile.write("%f " % p2x)
#         outfile.write("%f " % p2y)
#         outfile.write("%f " % p3x)
#         outfile.write("%f " % p3y)
#         outfile.write("%f " % p4x)
#         outfile.write("%f " % p4y)
#         outfile.write("%s " % label)
#         outfile.write("%d\n" % 0)



for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(out_dir, file_name)
    with open(input_path, 'r') as infile:
        # file_txts = infile.readlines()[1:]
        file_txts = infile.readlines()

    with open(output_path, 'w') as outfile:
        for file_ in file_txts:
            # 去除行尾换行符后再分割
            bbox_info = file_.strip().split()
            print(bbox_info)
            
            # 确保转换的每个值是浮动数
            obb = np.array(bbox_info[:5], dtype=float)
            obb[2] = obb[2] * 2
            obb[3] = obb[3] * 2
            print(obb)
            label = 'star'
            p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = xywhaTodota(obb)
            outfile.write("%f " % p1x)
            outfile.write("%f " % p1y)
            outfile.write("%f " % p2x)
            outfile.write("%f " % p2y)
            outfile.write("%f " % p3x)
            outfile.write("%f " % p3y)
            outfile.write("%f " % p4x)
            outfile.write("%f " % p4y)
            outfile.write("%s " % label)
            outfile.write("%d\n" % 0)


    # print(f"Converted {len(x)} bboxes to {output_path}")