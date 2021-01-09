from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
from heapq import *
import cv2 as cv
import os


# Load picture and detect edges
img = io.imread("imgs/c1.png", as_gray=True)
img = img > 0

h = 14
se = sk.morphology.disk(h//2)
print(se)
img = sk.morphology.binary_opening(img, se)
img = sk.morphology.binary_erosion(img, se)
img = sk.morphology.binary_erosion(img)
se = sk.morphology.disk(h//4)
img = sk.morphology.binary_dilation(img, se)

#img = sk.morphology.thin(img)
show_images([img])
