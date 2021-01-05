from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
import cv2 as cv
import os

img = io.imread("chords_1.png", as_gray=True)

show_images([img])
