import numpy as np
import os
import cv2 



def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi 
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a


file_root = "data/data_train_star/galaxy_lognorm/test/annos"
file_list = os.listdir(file_root)
for file in file_list: 
    
    gt_ft = open(os.path.join('data/data_train_star/galaxy_lognorm/test/annos_norm',file),'w')
    with open (os.path.join(file_root, file), 'r+') as rd:
        file_txts = rd.readlines()
        for line in file_txts:
            bbox_info = line.split()
            poly = np.array(bbox_info[:8], dtype=np.float32)
            label = bbox_info[8]
            x, y, a, b, ag = poly2obb_np_le90(poly)
            gt_ft.write("%f " % x)
            gt_ft.write("%f " % y)
            gt_ft.write("%f " % a)
            gt_ft.write("%f " % b)
            gt_ft.write("%f " % ag)
            gt_ft.write("%s\n" % label)
            # gt_ft.write("")
            # gt_ft.write("star\n")
            