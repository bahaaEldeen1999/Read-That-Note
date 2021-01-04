from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
from heapq import *
import cv2 as cv
import os


def binarize(img, block_size=35):
    t = sk.filters.threshold_local(img, block_size, offset=10)
    img_b = img < t
    return img_b


def runs_of_ones_array(bits):
    # print(np.invert(bits))
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts


def verticalRunLength(img):
    # white runs
    arr = []
    cdef int i
    for i in range(0, img.shape[1]):
        a = runs_of_ones_array(img[:, i])
        for x in a:
            arr.append(x)
    # print(arr)
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


def calcWeightFunc(p, q, c1, c2):
    if p == q:
        if p or q:
            return c1
        return c2
    if p or q:
        return 2*c1
    return 2*c2


def constructGraph(img):
    di = [0, 0, 1, -1, 1, -1, 1, -1]
    dj = [1, -1, 0, 0, 1, -1, -1, 1]
    # graph = []
    n = img.shape[0]*img.shape[1]
    graph = {}
    cdef int ni
    cdef int nj
    cdef int i
    cdef int j
    cdef int k
    cdef int indx2
    cdef int w
    cdef int indx = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(8):
                ni = i + di[k]
                nj = j + dj[k]
                # print("k "+str(k))
                if ni >= 0 and ni < img.shape[0] and nj >= 0 and nj < img.shape[1]:
                    w = calcWeightFunc(img[i][j], img[ni][nj], 6, 12)
                    indx2 = ni*img.shape[1]+nj
                    if indx in graph:
                        graph[indx].append({'node': indx2, 'w': w})
                    else:
                        graph[indx] = [{'node': indx2, 'w': w}]

            indx += 1
    g = []
    for i in range(n):
        g.append(graph[i])
    # print(g)
    return g


def relaxEdges(v, d, distance, priority_q):
    distance[v] = d
    heapq.heappush(priority_q, [d, v])


def dijkstra(img, graph, r, p1, p2):
    dis = []
    p = []
    cdef int indxStart = r*img.shape[1]-p1
    cdef int n = len(graph)  # 2*img.shape[0]*img.shape[1]
    cdef INF = 100000000
    cdef int i
    cdef int w
    cdef int edge_len
    # for i in range(n):
    #     dis.append(INF)
    #     p.append(-1)
    # dis[indxStart] = 0
    # p[indxStart] = indxStart
    # priority_q = []
    # finished = set()
    # relaxEdges(indxStart, 0, dis, priority_q)
    # cdef int indx = 0
    # while len(priority_q) and len(finished) != n:
    #     _, u = heapq.heappop(priority_q)
    #     finished.add(u)
    #     for edge in graph[u]:
    #         if edge['node']-p1 >= 0 and edge['node']-p1 < n and dis[edge['node']-p1] > dis[u]+edge['w']:
    #             relaxEdges(edge['node']-p1, dis[u]+edge['w'], dis, priority_q)
    #             p[edge['node']-p1] = u
    # print("N "+str(n))
    A = [None]*n
    p = [-1]*n
    p[indxStart] = indxStart
    queue = [(0, indxStart)]
    while queue:
        path_len, v = heappop(queue)
        if A[v] is None:  # v is unvisited
            A[v] = path_len
            for edge in graph[v]:
                w = edge['node']-p1
                edge_len = edge['w']
                if w >= 0 and w < n and A[w] is None:
                    heappush(queue, (path_len + edge_len, w))
                    p[w] = v
    return p


def getPath(img, p, r, p1):
    cdef int indxStart = r*img.shape[1]-p1
    path = []
    cdef int v = indxStart+img.shape[1]-1
    while not(v == indxStart):
        # print(v)
        path.append(v)
        v = p[v]
        if v == -1:
            return path
    path.append(indxStart)
    return path


def getWhitePercent(img, path, p1):
    cdef float b = 0
    cdef float w = 0
    cdef float x = 0
    cdef int i
    cdef int u
    cdef int c
    cdef int r
    for i in path:
        # print(i)
        u = i + p1
        r = u//img.shape[1]
        c = u % img.shape[1]
        if img[r][c] == True:
            w += 1
        else:
            b += 1
        x += 1
    return w/x


def trimPath(img, path, white_run, p1):
    newPath = []
    cdef int lastRun = 0
    cdef int startIndx = 0
    cdef int endIndx = 0
    cdef int i
    cdef int c
    cdef int r
    for i in range(len(path)):
        path[i] += p1
        r = path[i]//img.shape[1]
        c = path[i] % img.shape[1]
        r = min(img.shape[0]-1, r)
        c = min(img.shape[1]-1, c)
        if img[r][c] == True:
            lastRun += 1
        else:
            lastRun = 0
        if lastRun >= white_run:
            startIndx = i-lastRun
            break
    lastRun = 0
    for i in range(len(path)-1, -1, -1):
        path[i] += p1
        r = path[i]//img.shape[1]
        c = path[i] % img.shape[1]
        r = min(img.shape[0]-1, r)
        c = min(img.shape[1]-1, c)
        if img[r][c] == True:
            lastRun += 1
        else:
            lastRun = 0
        if lastRun >= white_run:
            endIndx = i+lastRun
            break
    for i in range(startIndx, endIndx):
        newPath.append(path[i])
    return newPath


def staffLineDetection(img, staff_space, staff_height):
    cdef float white_perc = 0.5
    graph = constructGraph(img)
    newImg = np.copy(img)
    newImg2 = np.zeros(img.shape)
    staff_lines = []
    cdef int strip_height = 2 * staff_space
    print("strip hegiht "+str(strip_height))
    cdef int i
    cdef int p1
    cdef int p2
    cdef int u
    cdef int j
    cdef int x
    for i in range(img.shape[0]):
        # print("row "+str(i))
        p1 = max(0, i-strip_height)
        # p1 = 0
        p2 = min(img.shape[0]-1, i+strip_height)
        # print("P1 "+str(p1)+" P2 "+str(p2))
        # p2 = 1
        # print(p1*img.shape[1], (p2+1) * img.shape[1])
        p = dijkstra(img, graph[p1*img.shape[1]:(p2+1)
                                * img.shape[1]], i, p1*img.shape[1], p2)
        # print(p)
        # print("finished dijksra")
        path = getPath(img, p, i, p1*img.shape[1])
        # print("finished get path")
        if getWhitePercent(img, path, p1*img.shape[1]) < white_perc:
            continue
        # staffLines.append(path)
        path = trimPath(img, path, strip_height, p1*img.shape[1])
        p1 = p1*img.shape[1]
        for x in path:
            u = x + p1
            r = u//img.shape[1]
            c = u % img.shape[1]
            r = min(img.shape[0]-1, r)
            c = min(img.shape[1]-1, c)
            # r1 = r-staff_height//2-1
            # r2 = r+staff_height//2+1
            # d = True
            # if (r1 >= 0 and img[r1][c] == True) or (r2 < img.shape[0] and img[r2][c] == True):
            #     d = False
            # if not d:
            #     continue
            for j in range(max(0, r-staff_height), min(img.shape[0], r+staff_height)):
                newImg[j][c] = False
                # newImg2[j][c] = True
        staff_lines.append(i)
        # print("white line ")
    # show_images([newImg2])
    return newImg, staff_lines


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


def lineSegmentationBasedOnStaves(img, staff_lines):
    blocks = np.zeros(img.shape)
    for x in staff_lines:
        blocks[max(0, x-5):min(img.shape[0]-1, x+5), :] = True
    newImgs = []
    cdef int i
    cdef int j
    i = 0
    j = 0
    while i < blocks.shape[0]:
        if blocks[i][0] == True:
            j = i+1
            while j < blocks.shape[0] and blocks[j][0] == True:
                j += 1
            print("i "+str(i)+" j "+str(j))
            newImgs.append(img[i:j, :])
            i = j - 1
        i += 1

    return newImgs


def charachterSegmentation(img_o, t):
    # img = sk.transform.resize(
    #     img, output_shape=(img.shape[0]*2, img.shape[1]*4), mode="constant", cval=0)
    # img = img > 0
    # print(img.shape)
    # show_images([img])
    # se = np.ones((1, 4))
    img = np.copy(img_o)
    # img = sk.morphology.binary_closing(img)
    # img_n = img_n ^ img
    # show_images([img_n])
    vertical_hist = np.sum(img, axis=0, keepdims=True)
    # print(vertical_hist)
    chars = []
    chars1 = []
    i = 0
    while i < img.shape[1]:
        if vertical_hist[0][i] > t:
            j = i + 1
            while j < img.shape[1] and vertical_hist[0][j] > t:
                j += 1
            # print("i "+str(i)+" j "+str(j))
            # chars1.append(img[:, i:j])
            c1 = max(0, i-2)
            c2 = min(img.shape[1], j+2)
            chars.append([0, img.shape[0], c1, c2])
            i = j - 1
        i += 1
    # show_images(chars1)
    return chars


def removeMusicalNotes(img, T_LEN):
    newImg = np.zeros(img.shape)
    cdef int k = 0
    cdef int i = 0
    cdef int j = 0
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


def extractCircleNotes(img, staff_height):
    # print(staff_height)
    newImg = np.copy(img)
    # se = np.ones(((staff_height)*2, (staff_height)*2))
    # newImg = sk.morphology.binary_dilation(newImg, se)
    se = np.ones(((staff_height+1)*2, (staff_height+1)*2))
    newImg = sk.morphology.binary_opening(newImg, se)
    # newImg = sk.morphology.binary_erosion(newImg, se)
    return newImg


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


def runCode():
    cdef int T_LEN
    folder = 'easy'
    try:
        os.mkdir(folder + "Out")
    except:
        pass
    for filename in os.listdir(folder):
        print("file "+str(filename))
        img = sk.io.imread(os.path.join(folder, filename), as_gray=True)
        # img = sk.transform.resize(img, (200, 200))
        # show_images([img])
        # img = sk.transform.resize(img, (500, 600))
        img = deskew(img)
        # show_images([img])
        img = img.astype(np.float64) / np.max(img)
        img = 255 * img
        img = img.astype(np.uint8)
        img = binarize(img, 101)
        # img_o = np.copy(img)
        # show_images([img_o])
        img = sk.morphology.binary_dilation(img)
        staff_height, staff_space = verticalRunLength(img)
        T_LEN = min(2*staff_height, staff_height+staff_space)
        ########
        img_1 = extractCircleNotes(img, staff_height)
        img_1 = img_1 > 0
        # show_images([img_1])
        ##########
        #####
        lines = classicLineSegmentation(img, staff_space)
        ######
        print(img.shape)
        # removed_staff, staff_lines = staffLineDetection(img, staff_space, staff_height)
        # show_images([removed_staff])
        removed_staff = removeMusicalNotes(img, T_LEN)
        removed_staff = removed_staff > 0
        removed_staff = removed_staff | img_1
        # show_images([removed_staff])
        # io.imsave("outputE/"+filename+"_removed_stadd.png",
        # sk.img_as_uint(removed_staff))
        se = np.ones((staff_height*2+1, staff_height*2+1))
        removed_staff_d = np.copy(removed_staff)
        removed_staff_d = sk.morphology.binary_dilation(removed_staff_d, se)
        # se = np.ones((1, staff_height+1))
        # removed_staff = sk.morphology.binary_dilation(removed_staff_d, se)
        # show_images([removed_staff])
        # removed_staff = sk.morphology.binary_dilation(removed_staff)
        i = 0
        try:
            os.mkdir(folder + "Out/"+filename+"_img")
        except:
            pass
        for line in lines:
            try:
                i += 1
                chars = charachterSegmentation(
                    removed_staff[line[0]:line[1], line[2]:line[3]], 0*staff_height*2)
                try:
                    os.mkdir(folder + "Out/"+filename +
                             "_img/"+filename+"_lines"+str(i))
                except:
                    pass
                j = 0
                for char in chars:
                    # print(char)
                    # show_images(
                    #     [removed_staff[line[0]:line[1], line[2]:line[3]][char[0]:char[1], char[2]:char[3]]])
                    io.imsave(folder + "Out/"+filename+"_img/"+filename+"_lines"+str(i)+"/"+filename+"_line_"+str(i) + "_char_"+str(
                        j)+".png", sk.img_as_uint(removed_staff[line[0]:line[1], line[2]:line[3]][char[0]:char[1], char[2]:char[3]]))
                    j += 1
            except:
                continue
        # io.imsave(folder+"Out/"+filename+"_removed_stadd_closing.png",sk.img_as_uint(removed_staff))
        # show_images([removed_staff])
        # staffs = lineSegmentation(removed_staff, staff_lines)

        # chars = charachterSegmentation(staffs[0], 0)
        print("done "+filename)
        # io.imsave("output/test7_removed_stadd_closing.png",removed_staff)
        # show_images(chars)
