from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
from heapq import *
import cv2 as cv

'''
convert image of any type to uint 8 byte
'''


def convertImgToUINT8(img_o):
    img = np.copy(img_o)
    img = img.astype(np.float64) / np.max(img)
    img = 255 * img
    img = img.astype(np.uint8)
    return img


'''
convert gray scale image to binary image 
'''


def binarize(img, block_size=101):
    t = sk.filters.threshold_local(img, block_size, offset=10)
    img_b = img < t
    return img_b


'''
deskew image to be horizontal lines
'''


def deskew(original_img):
    img = np.copy((original_img))
    # Canny
    imgCanny = sk.feature.canny(img, sigma=1.5)
    thresh = sk.filters.threshold_otsu(imgCanny)
    imgCanny = (imgCanny >= thresh)

    # Apply Hough Transform
    # Generates a list of 360 Radian degrees (-pi/2, pi/2)
    angleSet = np.linspace(-np.pi, np.pi, 1440)
    houghArr, theta, dis = sk.transform.hough_line(imgCanny, angleSet)

    flatIdx = np.argmax(houghArr)
    bestTheta = (flatIdx % theta.shape[0])
    bestTheta = angleSet[bestTheta]
    bestDis = np.int32(np.floor(flatIdx / theta.shape[0]))
    bestDis = dis[bestDis]

    # Rotate
    thetaRotateDeg = (bestTheta*180)/np.pi
    if thetaRotateDeg > 0:
        thetaRotateDeg = thetaRotateDeg - 90
    else:
        thetaRotateDeg = thetaRotateDeg + 90

    imgRotated = (sk.transform.rotate(
        img, thetaRotateDeg, resize=True, mode='constant', cval=1))
    return imgRotated


'''
run lenght encoding on number of ones in array of booleans/bits
'''


def runs_of_ones_array(bits):
    bounded = np.hstack(([0], bits, [0]))
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts


'''
extract staff height and staff space based on run lenght encoding of white bits in binary representation of each column in the image 
----based on what the papers was doing
'''


def verticalRunLength(img):
    # white runs
    arr = []
    cdef int i
    for i in range(0, img.shape[1]):
        a = runs_of_ones_array(img[:, i])
        for x in a:
            arr.append(x)
    counts = np.bincount(arr)
    staff_height = np.argmax(counts)
    # black runs
    arr = []
    for i in range(0, img.shape[1]):
        a = runs_of_ones_array(np.invert(img[:, i]))
        for x in a:
            arr.append(x)
    # print(arr)
    counts = np.bincount(arr)
    staff_space = np.argmax(counts)
    return staff_height, staff_space


'''
segment each staff by itself
'''


def classicLineSegmentation(img, staff_space=0):
    org = np.copy(img)
    lines = []
    se = np.ones((staff_space+5, 2))
    img = sk.morphology.binary_dilation(img, se)
    horz_hist = np.sum(img, axis=1)
    t = 0.25
    i = 0
    j = 0
    while i < img.shape[0]:
        if horz_hist[i]/img.shape[1] >= t:
            j = i + 1
            while j < img.shape[0] and horz_hist[j]/img.shape[1] >= t:
                j += 1
            # print("i "+str(i)+" j "+str(j))
            r0 = max(0, i-staff_space+5)
            r1 = min(img.shape[0], j+staff_space+5)
            lines.append([r0, r1, 0, img.shape[1]])
            i = j - 1
        i += 1
    # show_images(lines)
    return lines


'''
remove staff lines from binary image 
'''


def extractMusicalNotes(img, T_LEN):
    newImg = np.zeros(img.shape)
    for i in range(0, img.shape[1]):
        arr = runs_of_ones_array(img[:, i])
        # print(arr)
        k = 0
        j = 0
        while j < img.shape[0]:
            if img[j][i] == True:
                if arr[k] > T_LEN:
                    for x in range(0, arr[k]):
                        newImg[j][i] = True
                        j += 1
                else:
                    j += arr[k]-1
                k += 1
            j += 1
    return newImg


'''
remove musical notes from staff lines
'''


def removeMusicalNotes(img, T_LEN):
    newImg = np.copy(img)
    for i in range(0, img.shape[1]):
        arr = runs_of_ones_array(img[:, i])
        # print(arr)
        k = 0
        j = 0
        while j < img.shape[0]:
            if img[j][i] == True:
                if arr[k] > T_LEN:
                    for x in range(0, arr[k]):
                        newImg[j][i] = False
                        j += 1
                else:
                    j += arr[k]-1
                k += 1
            j += 1
    return newImg


'''
restore staff liens after notes removal
'''


def restoreStaffLines(img, T_LEN, img_o):
    newImg = np.copy(img)
    for i in range(0, img.shape[1]):
        arr = runs_of_ones_array(img_o[:, i])
        # print(arr)
        k = 0
        j = 0
        while j < img.shape[0]:
            if img_o[j][i] == True:
                if arr[k] > T_LEN:
                    for x in range(0, arr[k]):
                        try:
                            newImg[j][i] = False
                            if np.sum(img[j, 0:i])+np.sum(img[j, i:img.shape[1]]) >= 0.1*img.shape[1]:
                                newImg[j][i] = True
                        except:
                            pass
                        j += 1
                else:
                    j += arr[k]-1
                k += 1
            j += 1
    return newImg


'''
fix restored staff lines by connecting broken lines 
'''


def fixStaffLines(staff_lines, staff_height, staff_space, img_o):
    img = np.copy(staff_lines)
    patch_height = 100
    patch_width = staff_lines.shape[1]//15
    ph = int(img.shape[0]/patch_height)
    pw = int(img.shape[1]/patch_width)
    for i in range(ph):
        for j in range(pw):
            patch = img[i*patch_height: (i+1)*patch_height,
                        j*patch_width: (j+1)*patch_width]
            for k in range(patch.shape[0]):
                x = np.sum(patch[k, :])
                if x >= 0.2*patch.shape[1]:
                    patch[k, :] = img_o[i*patch_height: (
                        i+1)*patch_height, j*patch_width: (j+1)*patch_width][k, :]
    return img
