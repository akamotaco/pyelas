#!/usr/bin/env python

import pathlib as pl
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

from elas import *

def convert_gray(img_rgb: np.ndarray, gamma: float = 1, rgb_to_bgr: bool = True) -> np.ndarray:
    # https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2] # r, g, b
    if rgb_to_bgr:
        B, G, R = R, G, B

    return 0.299*R + 0.587*G + 0.114*B

    Y = R**gamma*0.2126 + G**gamma*0.7152 + B**gamma*0.0722
    L = 166 * Y **(1/3) - 16
    return Y # L

def fill_gap(disparity_map: np.ndarray, method: str = 'linear') -> np.ndarray:
    d_map = disparity_map.copy()

    if method == 'linear':
        def fill_func(d_map, idx_range):
            blending_list = np.linspace(0,1,len(idx_range))
            start_idx = idx_range[0]
            start_value, end_value = d_map[idx_range[0]], d_map[idx_range[-1]]
            for idx in idx_range:
                blending = blending_list[idx - start_idx]
                d_map[idx] = (1-blending)*start_value + blending*end_value
    else:
        raise Exception(f'{method} is unknown error')

    height, width = d_map.shape
    for y in range(height):
        start_point = None
        value_prev = None
        for x in range(width):
            value = d_map[y, x]
            if value == 0:
                if start_point is None and value_prev is not None and value_prev > 0:
                    start_point = x-1
                else:
                    pass
            else:
                if start_point is not None: # fill
                    end_point = x
                    idx_range = list(range(start_point, end_point + 1))
                    fill_func(d_map[y], idx_range)
                    start_point = None
            value_prev = value
    return d_map

def point_cloud_from_disparity_map(disparity_map: np.ndarray, max_z: float = 1000.0) -> np.ndarray:
    # fujifilm 3d w3 (https://asset.fujifilm.com/www/jp/files/2019-12/c46a5d7db9151ab73830c0b42870079d/ff_rd057_008_en.pdf)
    focal_length = 6.3 # 6.3 (mm)
    baseline_length = 75 # 75 (mm)
    horizontal_pixels = disparity_map.shape[1] #3648 # L-size
    sensor_pixel_mm = 6.17 # (mm) # CCD = 1/2.3 # inches  (6.17mm x 4.55mm) 

    pixel_pitch = sensor_pixel_mm / horizontal_pixels

    Cx, Cy = disparity_map.shape[1]/2, disparity_map.shape[0]/2

#region parallel setup
    print('special case: parallel setup')
    # https://stackoverflow.com/questions/50297459/how-to-compute-the-true-depth-given-a-disparity-map-of-rectified-images
    height, width = disparity_map.shape
    point_list = []
    for y in range(height):
        for x in range(width):
            disparity = disparity_map[y, x] * pixel_pitch
            if disparity == 0:
                X = x
                Y = y
                Z = max_z
            else:
                X = baseline_length * (x-Cx) *  pixel_pitch / disparity
                Y = baseline_length * (y-Cy) *  pixel_pitch
                Z = (baseline_length * focal_length) / disparity
                if Z > max_z:
                    Z = max_z
            point_list.append([X, Y, Z])

    point_list = np.array([pnt for pnt in point_list if pnt[2] < max_z])
    return point_list
#endregion

#     print('general case: not completed')

#     Cx, Cy = None, None
#     f = None
#     a, b = None, None

#     Cx, Cy = disparity_map.shape[1]/2, disparity_map.shape[0]/2

#     f = focal_length
#     Tx = baseline_length
#     Cx_prime = Cx # https://stackoverrun.com/ko/q/7515441 # rectification?
#     # Cx_prime = Cx - pixel_pitch

#     a, b = -1 / Tx, (Cx - Cx_prime) / Tx

#     Q = np.zeros((4,4))
#     Q[0, 0] = Q[1, 1] = 1
#     Q[0, 3] = -Cx
#     Q[1, 3] = -Cy
#     Q[2, 3] = f
#     Q[3, 2] = a
#     Q[3, 3] = b

#     # height, width = disparity_map.shape
#     # tmp = np.zeros(disparity_map.shape + (3,))
#     # for y in range(height):
#     #     for x in range(width):
#     #         disp = disparity_map[y, x]
#     #         if disp == 0:
#     #             disp = 0.001
#     #         d = baseline_length
#     #         tmp[y, x][0] = d*x / disp
#     #         tmp[y, x][1] = y
#     #         tmp[y, x][2] = d*f / disp
#     # return tmp.reshape(-1,3)

#     Q_ = np.float32([[1,0,0,-Cx],
#                 [0,-1,0,Cy],
#                 [0,0,0,-focal_length],
#                 [0,0,1,0]])
    
#     # https://azerdark.wordpress.com/tag/fujifilm-w3-3d-camera/
#     Q__ = np.array([1.,0.,0.,-2.8327271270751953e+002,0.,1.,0.,-1.5946473121643066e+002,
#             0.,0.,0.,1.0546290540664800e+003,0.,0.,-2.3597727835463600e-001,
#             -1.7014427703509563e+000]).reshape(4,4)

#     # https://answers.opencv.org/question/4379/from-3d-point-cloud-to-disparity-map/
#     image_matrix = np.ones((np.prod(disparity_map.shape),) + (4,))
#     height, width = disparity_map.shape
#     for y in range(height):
#         for x in range(width):
#             Ix, Iy = x, y
#             d = disparity_map[y, x]
#             idx = y*width + x
#             image_matrix[idx][0] = Ix
#             image_matrix[idx][1] = Iy
#             image_matrix[idx][2] = d
    
#     estimation = image_matrix@Q.T # (Q@image_matrix.T).T
#     # es = cv2.reprojectImageTo3D(disparity_map, Q)

#     w_axis = estimation[:, [3]]
#     w_axis[w_axis==0] = 0.5
#     estimation[:, [3]] = w_axis

#     estimation_3d = estimation[:,:3] / estimation[:, [3]] # x/w, y/w, z/w

# #region reconstruction
#     recon = []
#     for pnt in estimation_3d:
#         X_, Y_, Z_ = pnt
#         d = (f - Z_ * b) / (Z_ * a)
#         Ix = X_ * (d * a + b) + Cx
#         Iy = Y_  * (d * a + b) + Cy
#         recon.append([Ix, Iy, d])
#     recon = np.array(recon)

#     recon_disparity_map = np.zeros(disparity_map.shape)
#     for info in recon:
#         x, y, d = info
#         x, y = x.round().astype(int), y.round().astype(int)
#         recon_disparity_map[y, x] = d
# #endregion

#     from IPython import embed;embed()

#     return estimation_3d

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage ./demo.py image1 image2')
        sys.exit(0)

    image1 = pl.Path(sys.argv[1])
    image2 = pl.Path(sys.argv[2])
    assert image1.exists() and image2.exists(), "Image files don't exist"

    im_left = convert_gray(imread(str(image1))).round().astype(np.uint8)
    im_right = convert_gray(imread(str(image2))).round().astype(np.uint8)
    # im1 = cv2.cvtColor(imread(str(image1)),cv2.COLOR_RGB2GRAY)
    # im2 = cv2.cvtColor(imread(str(image2)),cv2.COLOR_RGB2GRAY)

    d1 = np.empty_like(im_left, dtype=np.float32)
    d2 = np.empty_like(im_right, dtype=np.float32)

    params = Elas_parameters() # struct parameters in elas.h
    params.postprocess_only_left = True
    elas = Elas(params)
    elas.process_stereo(im_left, im_right, d1, d2)

    d1[d1<0] = 0
    d2[d2<0] = 0

    # fig, (ax1, ax2) = plt.subplots(1,2)
    # ax1.set_title('d1')
    # ax1.imshow(d1)
    # ax2.set_title('d2')
    # ax2.imshow(d2)
    # plt.tight_layout()
    # plt.show()

    d1_1 = fill_gap(d1)
    # d2_1 = fill_gap(d2)
    # plt.imshow(d1_1)
    # plt.show()

    pc = point_cloud_from_disparity_map(d1_1)

    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # disparity = stereo.compute(im_left,im_right)

    # im_left_color = imread(str(image1))
    # im_left_color = im_left_color[:,:,:3]/255

    # colored_vertex = []
    # for vertex in pc:
    #     x, y = vertex[0].astype(int), vertex[1].astype(int)
    #     color = im_left_color[y, x]
    #     colored_vertex.append(np.concatenate([vertex, color]))
    # colored_vertex = np.array(colored_vertex)

    from IPython import embed;embed()

    from wavefront import Wavefront
    obj = Wavefront()
    obj.objects['__temp__']['v'] = pc # colored_vertex
    obj.save('tmp.obj')
