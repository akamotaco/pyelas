#!/usr/bin/env python

import pathlib as pl
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

from elas import *

def convert_gray(img_rgb: np.ndarray, gamma: float = 1, rgb_to_bgr: bool = True):
    # https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2] # r, g, b
    if rgb_to_bgr:
        B, G, R = R, G, B

    return 0.299*R + 0.587*G + 0.114*B

    Y = R**gamma*0.2126 + G**gamma*0.7152 + B**gamma*0.0722
    L = 166 * Y **(1/3) - 16
    return Y # L

if len(sys.argv) < 3:
    print('Usage ./demo.py image1 image2')
    sys.exit(0)

image1 = pl.Path(sys.argv[1])
image2 = pl.Path(sys.argv[2])
assert image1.exists() and image2.exists(), "Image files don't exist"

im1 = convert_gray(imread(str(image1))).round().astype(np.uint8)
im2 = convert_gray(imread(str(image2))).round().astype(np.uint8)
# im1 = cv2.cvtColor(imread(str(image1)),cv2.COLOR_RGB2GRAY)
# im2 = cv2.cvtColor(imread(str(image2)),cv2.COLOR_RGB2GRAY)

d1 = np.empty_like(im1, dtype=np.float32)
d2 = np.empty_like(im2, dtype=np.float32)

params = Elas_parameters() # struct parameters in elas.h
params.postprocess_only_left = True
elas = Elas(params)
elas.process_stereo(im1, im2, d1, d2)

d1[d1<0] = 0
d2[d2<0] = 0

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_title('d1')
ax1.imshow(d1)
ax2.set_title('d2')
ax2.imshow(d2)
plt.tight_layout()
plt.show()
