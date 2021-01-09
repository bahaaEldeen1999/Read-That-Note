from matplotlib.pyplot import show
from commonfunctions import *
from preprocessing import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
from heapq import *
import cv2 as cv

'''
get note head charachter basedon its position
'''


def getFlatHeadNotePos(staff_lines, note, staff_space, charPos, staff_height, img_o=None):
    if charPos[3]-charPos[2] < staff_space:
        return [-1]
    img = np.copy(note)
    # show_images([img])
    s_c = np.copy(staff_lines)
    n_c = np.copy(note)
    s_c = s_c > 0
    n_c = n_c > 0
    s_c[charPos[0]:charPos[1], charPos[2]:charPos[3]
        ] = s_c[charPos[0]:charPos[1], charPos[2]:charPos[3]] | n_c
    err = staff_space//8
    # edges = np.copy(note)
    # edges = edges.astype(int)*255
    # show_images([edges])
    # se = sk.morphology.disk(staff_space//16)
    # img = sk.morphology.binary_closing(img, se)
    # img = sp.ndimage.morphology.binary_fill_holes(img)

    # show_images([img])
    img = sk.morphology.binary_closing(img)
    #img = sk.morphology.binary_closing(img)
    se = sk.morphology.disk(staff_space//2-1)
    img = sk.morphology.binary_opening(img, se)
    # show_images([img])
    #se = sk.morphology.disk((staff_space//3))
    se = sk.morphology.disk(staff_space//4)
    img = sk.morphology.binary_erosion(img)
    img = sk.morphology.binary_dilation(img)
    se = sk.morphology.disk(staff_space//2-1)
    img = sk.morphology.binary_erosion(img, se)
    #img = sk.morphology.binary_erosion(img)
    se = sk.morphology.disk(staff_space//8+1)
    img = sk.morphology.binary_dilation(img, se)
    img = sk.morphology.binary_erosion(img)
    # show_images([img])
    bounding_boxes = sk.measure.find_contours(img, 0.8)
    output = [charPos[2]]
    # print(len(bounding_boxes))
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = [np.min(box[:, 1]), np.max(
            box[:, 1]), np.min(box[:, 0]), np.max(box[:, 0])]
        ar = (Xmax-Xmin)/(Ymax-Ymin)
        if True:
            r0 = int(Ymin)
            r1 = int(Ymax)
            r0 = max(r0, 0)
            r1 = min(r1, staff_lines.shape[0])
            center = (r0+r1)//2
            center2 = staff_lines.shape[1]//2
            #print("center "+str(center))
            #x = np.copy(note)
            #x[center-staff_space//2:center+staff_space//2, :] = False
            # print(staff_lines[center-staff_height:center+staff_height, :])
            # show_images(
            #     [note, x, staff_lines[center-staff_space//2:center+staff_space//2, :]])
            # show_images(
            #     [s_c[center-staff_space//2:center+staff_space//2, :]])
            horz_hist = np.sum(
                staff_lines[center-staff_height:center+staff_height, :], axis=1)
            # print(horz_hist)
            maximum = np.sum(horz_hist)
            up_arr = runs_of_ones_array(
                staff_lines[0:center-staff_height//2, center2])
            up = 0
            t = staff_height//2
            for x in up_arr:
                if x >= t:
                    up += 1
            # print(maximum)
            if maximum < staff_lines.shape[1]:
                # print("space")
                if up == 0:
                    x = 0
                    i = center
                    while i < staff_lines.shape[0] and staff_lines[i, center2] == False:
                        x += 1
                        i += 1
                    if x < staff_space:
                        output.append("g2")
                    elif x < 1.5*staff_space:
                        output.append("a2")
                    else:
                        output.append("b2")
                elif up == 1:
                    output.append("e2")
                elif up == 2:
                    output.append("c2")
                elif up == 3:
                    output.append("a1")
                elif up == 4:
                    output.append("f1")
                else:
                    x = 0
                    i = center
                    while i >= 0 and staff_lines[i, center2] == False:
                        x += 1
                        i -= 1
                    if abs(x) < staff_space:
                        output.append("d1")
                    else:
                        output.append("c1")
            else:
                # print("line")
                if up == 5:
                    x = 0
                    i = center
                    while i >= 0 and staff_lines[i, center2] == False:
                        x += 1
                        i -= 1
                    if abs(x) < staff_space:
                        output.append("d1")
                    else:
                        output.append("c1")
                elif up == 1:
                    output.append("d2")
                elif up == 2:
                    output.append("b1")
                elif up == 3:
                    output.append("g1")
                elif up == 4:
                    output.append("e1")
                elif up == 0:
                    x = 0
                    i = center
                    while i < staff_lines.shape[0] and staff_lines[i, center2] == False:
                        x += 1
                        i += 1
                    # print("X "+str(x))
                    if x < staff_space:
                        output.append("f2")
                    elif x < 1.5*staff_space:
                        output.append("a2")
                    else:
                        output.append("b2")

    return output
