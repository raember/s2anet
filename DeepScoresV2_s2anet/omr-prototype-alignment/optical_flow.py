import numpy as np
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp


def optical_flow_merging(img, prototype):
    nr, nc = img.shape
    v, u = optical_flow_tvl1(img, prototype, attachment=10, tightness=0.3)
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    prototype_warp = warp(prototype, np.array([row_coords + v, col_coords + u]), mode='edge')
    return prototype_warp
