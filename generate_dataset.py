import os
import cv2
import scipy as sp
import matplotlib as mp
import numpy as np
import skimage as sk
from commonfunctions import *

'''
idx=0
items = os.listdir("beams")
for img_name in items:
    img = io.imread("beams/{}".format(img_name), as_gray=True)
    (rows, cols) = img.shape
    os.mkdir("beams/{}".format(str(idx)))
    print(img.shape)
    for i in range(1, 360):
        imgNew = img.copy()
        imgNew = sk.transform.rotate(imgNew, i, resize=True, mode='edge')
        plt.imsave("beams/{}/{}.png".format(str(idx), str(i)),
                   imgNew, cmap='gray')
    idx += 1

'''

idx = 0
for file_name in os.listdir("dataset_mixed2 (2)/dataset_mixed2/"):
    if file_name == 'dataset_mixed' or file_name == 't4':
        continue
    items = os.listdir(
        "dataset_mixed2 (2)/dataset_mixed2/{}/".format(file_name))
    for img_name in items:
        img = io.imread(
            "dataset_mixed2 (2)/dataset_mixed2/{}/{}".format(file_name, img_name), as_gray=True)
        img = (img <= sk.filters.threshold_otsu(img)).astype(int)
        plt.imsave("dataset_mixed2 (2)/dataset_mixed2/{}/{}".format(file_name,
                                                                    img_name), img, cmap='gray')
