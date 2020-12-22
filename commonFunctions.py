import numpy as np
import math
import cv2 as cv
import scipy as sp
from scipy.signal import convolve2d
from scipy import fftpack
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import bar

import skimage.io as io
from skimage.exposure import *
from skimage.color import rgb2gray, rgb2hsv, label2rgb
from skimage import data, img_as_float
from skimage.feature import *
from skimage.transform import *
from skimage.filters import *
from skimage.util import random_noise
from skimage.measure import *
from skimage.morphology import *
from skimage.draw import *

# Convolution:


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')
