from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
from heapq import *
import cv2

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
get frequency of start staffs in every row
'''


def get_lines_rows(img, T_LEN):
    row_start_freq = np.zeros((1, img.shape[0]+5))[0]
    row_starts = []

    for i in range(0, img.shape[1]):
        arr = runs_of_ones_array(img[:, i])
        k = 0
        j = 0
        while j < img.shape[0]:
            if img[j][i] == True:
                if arr[k] <= T_LEN + 2 and arr[k] >= T_LEN - 2:
                    row_start_freq[j] += 1
                    j += arr[k]-1
                else:
                    j += arr[k]

                k += 1
            j += 1

    max_freq_row_start = 0
    for r in row_start_freq:
        max_freq_row_start = max(max_freq_row_start, r)

    for i in range(len(row_start_freq)):
        # Approximately, if the row "i" is frequently treated as a starting of staffs with this ratio
        # by the most frequnt starting row, then consider it as a starting row of staffs.
        if row_start_freq[i]/max_freq_row_start >= 0.12:
            row_starts.append(i)
    return [row_starts, row_start_freq, max_freq_row_start]


'''
remove staff lines from binary image
'''


def extractMusicalNotes(img, T_LEN):
    staff_rows_starts, row_start_freq, max_freq_row_start = get_lines_rows(
        img, T_LEN)
    is_here = np.zeros((1, img.shape[0] + 10))[0]
    for x in staff_rows_starts:
        is_here[x] = 1
    newImg = np.zeros(img.shape)

    for i in range(0, img.shape[1]):
        arr = runs_of_ones_array(img[:, i])
        block_num = 0
        row = 0
        while row < img.shape[0]:
            if img[row][i] == True:

                found = False
                for idx in range(0, int(1.5*T_LEN)):
                    if row - idx >= 0 and row - idx < img.shape[0]:
                        # and row_start_freq[row] / max_freq_row_start >= 0.1:
                        if is_here[row - idx]:
                            found = True
                            jump = T_LEN
                            row += jump
                            arr[block_num] -= jump
                            arr[block_num] = max(arr[block_num], 0)
                            if arr[block_num] > 0:
                                block_num -= 1
                            break
                if found == False:
                    for item in range(arr[block_num]):
                        if row >= img.shape[0]:
                            break
                        newImg[row][i] = True
                        row += 1
                row -= 1
                block_num += 1
            row += 1
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


'''
get charachters corners from the staff line
'''


def char_seg(org_img):
    # show_images([org_img])
    img = np.copy(org_img)

    toshow = [img]

    labels = sk.measure.label(img, connectivity=1)
    lbl_num = np.max(labels[:, :])

    bounds = np.zeros((lbl_num+1, 4))  # [up, down, left, right]
    bounds[:, 0] = 99999999
    bounds[:, 2] = 99999999

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j]:
                bounds[labels[i, j]][0] = int(min(bounds[labels[i, j]][0], i))
                bounds[labels[i, j]][1] = int(max(bounds[labels[i, j]][1], i))
                bounds[labels[i, j]][2] = int(min(bounds[labels[i, j]][2], j))
                bounds[labels[i, j]][3] = int(max(bounds[labels[i, j]][3], j))

    only_char_arr = []

    for i in range(bounds.shape[0]):
        if bounds[i][0] == 99999999:
            continue
        cur = np.copy(labels[int(bounds[i][0]):int(
            bounds[i][1]+1), int(bounds[i][2]):int(bounds[i][3]+1)])
        cur = cur == i
        only_char_arr.append(cur)

    return [bounds, only_char_arr]


'''
extract filled note heads from image
'''


def extractCircleNotes(img_o, staff_space):
    img = np.copy(img_o)
    se = sk.morphology.disk(staff_space//2)
    img = sk.morphology.binary_opening(img, se)
    img = sk.morphology.binary_erosion(img, se)
    img = sk.morphology.binary_erosion(img)
    se = sk.morphology.disk(staff_space//4)
    img = sk.morphology.binary_dilation(img, se)
    return img


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
            r0 = int(max(0, i-staff_space*2-5))
            r1 = int(min(img.shape[0], j+staff_space*2+5))
            lines.append([r0, r1, 0, img.shape[1]])
            i = j - 1
        i += 1
    return lines
