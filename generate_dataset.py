import os
import cv2
import scipy as sp
import matplotlib as mp
import numpy as np
import skimage as sk
from commonfunctions import *


items = os.listdir("imgs")
for img_name in items:
    img = io.imread("imgs/{}".format(img_name), as_gray=True)
    label = img_name.split('.')[0]
    os.mkdir("imgs/{}".format(label))
    idx = 1
    for rotation in range(1, 360):
        for scale in range(0.1, 3, 0.1):
            imgNew = img.copy()
            imgNew = sk.transform.warp(
                imgNew, sk.transform.SimilarityTransform(scale=scale, rotation=rotation))
            plt.imsave("imgs/{}/{}{}.png".format(str(label), label, str(idx)),
                       imgNew, cmap='gray')
            idx += 1


# idx = 0
# imgNew = sk.transform.rotate(imgNew, i, resize=True, mode='edge')
# for file_name in os.listdir("dataset_mixed2 (2)/dataset_mixed2/"):
#     if file_name == 'dataset_mixed' or file_name == 't4':
#         continue
#     items = os.listdir(
#         "dataset_mixed2 (2)/dataset_mixed2/{}/".format(file_name))
#     for img_name in items:
#         img = io.imread(
#             "dataset_mixed2 (2)/dataset_mixed2/{}/{}".format(file_name, img_name), as_gray=True)
#         img = (img <= sk.filters.threshold_otsu(img)).astype(int)
#         plt.imsave("dataset_mixed2 (2)/dataset_mixed2/{}/{}".format(file_name,
#                                                                     img_name), img, cmap='gray')
