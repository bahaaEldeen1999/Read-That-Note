from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
from heapq import *
import cv2 as cv
import os


# Load picture and detect edges
img = io.imread("imgs/t6.jpg", as_gray=True)
img = img > 0
# img = img.astype(np.float64) / np.max(img)
# img = 255 * img
# img = img.astype(np.uint8)
h = 14
#se = np.ones((9, 9))
#img = sk.morphology.binary_erosion(img, se)
#img = sk.feature.canny(img)
show_images([img])
img = sp.ndimage.binary_fill_holes(img)
show_images([img])
# img = sp.ndimage.binary_fill_holes(img)
# img = sp.ndimage.binary_fill_holes(img)
# img = sp.ndimage.binary_fill_holes(img)
# img = sp.ndimage.binary_fill_holes(img)
# img = sp.ndimage.binary_fill_holes(img)
# img = sp.ndimage.binary_fill_holes(img)
# img = sp.ndimage.binary_fill_holes(img)
# img = sp.ndimage.binary_fill_holes(img)
# img = sp.ndimage.binary_fill_holes(img)

se = sk.morphology.disk(h//2)
# print(se)
img = sk.morphology.binary_opening(img, se)
img = sk.morphology.binary_erosion(img, se)
img = sk.morphology.binary_erosion(img)
se = sk.morphology.disk(h//4)
img = sk.morphology.binary_dilation(img, se)

#img = sk.morphology.thin(img)
show_images([img])
