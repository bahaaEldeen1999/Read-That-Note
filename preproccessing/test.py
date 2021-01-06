from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
from heapq import *
import cv2 as cv
import os


# Load picture and detect edges
img = io.imread("imgs/n4.png", as_gray=True)
img = img > 0
staff_height = 15
se = np.ones((staff_height, staff_height))
img = sk.morphology.binary_opening(img, se)
# show_images([img])
bounding_boxes = sk.measure.find_contours(img, 0.8)
# When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in img_with_boxes
newImg = np.zeros(img.shape)
print(len(bounding_boxes))
for box in bounding_boxes:
    # print(np.max(box[:,1]))
    #box = np.uint8(box)
    # print(box)
    [Xmin, Xmax, Ymin, Ymax] = [np.min(box[:, 1]), np.max(
        box[:, 1]), np.min(box[:, 0]), np.max(box[:, 0])]
    ar = (Xmax-Xmin)/(Ymax-Ymin)
    #ar = 1/ar
    # if ( ( ar<=3.5 and ar>= 2.5)):

    # print(ar)
    rr, cc = sk.draw.rectangle(start=(Ymin, Xmin), end=(
        Ymax, Xmax), shape=newImg.shape)

    rr = rr.astype(int)
    cc = cc.astype(int)
    newImg[rr, cc] = True

    #print(Xmin, Xmax, Ymin, Ymax)
    # print(rr)

show_images([img, newImg])
