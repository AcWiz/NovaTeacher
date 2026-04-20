import pandas as pd
import numpy as np
import os 


def xywhaTodota(rbboxes):
    rbboxes.astype(float)
    x = rbboxes[:, 0].reshape(-1, 1).astype(float)
    y = rbboxes[:, 1].reshape(-1, 1).astype(float)
    x = np.round(x).astype(float)
    y = np.round(y).astype(float)
    # x = np.round(x).astype(int)
    # y = np.round(y).astype(int)
    w = rbboxes[:, 2].reshape(-1, 1).astype(float) * 2
    h = rbboxes[:, 3].reshape(-1, 1).astype(float) * 2 
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


data_list_dir = os.listdir("")
 