from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
import heapq


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
    #graph = []
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
                #print("k "+str(k))
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
    for i in range(n):
        dis.append(INF)
        p.append(-1)
    dis[indxStart] = 0
    p[indxStart] = indxStart
    priority_q = []
    finished = set()
    relaxEdges(indxStart, 0, dis, priority_q)
    cdef int indx = 0
    while len(priority_q) and len(finished) != n:
        _, u = heapq.heappop(priority_q)
        finished.add(u)
        for edge in graph[u]:
            if edge['node']-p1 >= 0 and edge['node']-p1 < n and dis[edge['node']-p1] > dis[u]+edge['w']:
                relaxEdges(edge['node']-p1, dis[u]+edge['w'], dis, priority_q)
                p[edge['node']-p1] = u
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
    cdef float white_perc = 0.75
    graph = constructGraph(img)
    newImg = np.copy(img)
    cdef int strip_height = 2 * staff_space
    print("strip hegiht "+str(strip_height))
    cdef int i
    cdef int p1
    cdef int p2
    cdef int u
    cdef int j
    cdef int x
    for i in range(img.shape[0]):
        #print("row "+str(i))
        p1 = max(0, i-strip_height)
        #p1 = 0
        p2 = min(img.shape[0]-1, i+strip_height)
        #print("P1 "+str(p1)+" P2 "+str(p2))
        # p2 = 1
        #print(p1*img.shape[1], (p2+1) * img.shape[1])
        p = dijkstra(img, graph[p1*img.shape[1]:(p2+1)
                                * img.shape[1]], i, p1*img.shape[1], p2)
        # print(p)
        #print("finished dijksra")
        path = getPath(img, p, i, p1*img.shape[1])
        #print("finished get path")
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
            r1 = r-staff_height-2
            r2 = r+staff_height+2
            d = True
            if (r1 >= 0 and img[r1][c] == True) or (r2 < img.shape[0] and img[r2][c] == True):
                d = False
            if not d:
                continue
            for j in range(max(0, r-staff_height), min(img.shape[0], r+staff_height)):
                newImg[j][c] = False
        print("white line ")
    return newImg


def deskewImg(img):
    imgCanny = sk.feature.canny(img, sigma=.9)
    imgCanny = binarize(img)
    # Generates a list of 360 Radian degrees (-pi/2, pi/2)
    angleSet = np.linspace(-np.pi, np.pi, 1440)
    houghArr, theta, dis = sk.transform.hough_line(imgCanny, angleSet)
    flatIdx = np.argmax(houghArr)
    bestTheta = (flatIdx % theta.shape[0])
    bestTheta = angleSet[bestTheta]
    bestDis = np.int32(np.floor(flatIdx / theta.shape[0]))
    bestDis = dis[bestDis]
    bestLine = np.zeros(imgCanny.shape)
    if bestTheta == 0:
        for y in range(imgCanny.shape[0]):
            bestLine[y, bestDis] = 1
    else:
        for x in range(bestLine.shape[1]):
            y = (bestDis - x*np.cos(bestTheta))/np.sin(bestTheta)
            y = np.int32(np.floor(y))
            if y < bestLine.shape[0] and y >= 0:
                bestLine[y, x] = 1
            else:
                pass
    lineAboveImgCanny = np.maximum(imgCanny, bestLine)
    thetaRotateDeg = (bestTheta*180)/np.pi
    if thetaRotateDeg > 0:
        thetaRotateDeg = thetaRotateDeg - 90
    else:
        thetaRotateDeg = thetaRotateDeg + 90
    imgRotated = sk.transform.rotate(img, thetaRotateDeg)
    imgRotated = imgRotated > 0
    return imgRotated


def runCode():
    img = sk.io.imread('imgs/test8.jpg', as_gray=True)
    img = sk.transform.resize(img, (400, 400))
    img = img.astype(np.float64) / np.max(img)
    img = 255 * img
    img = img.astype(np.uint8)
    img = binarize(img)
    img = deskewImg(img)
    # show_images([img])
    staff_height, staff_space = verticalRunLength(img)
    cdef int T_LEN = min(2*staff_height, staff_height+staff_space)
    print(img.shape)
    removed_staff = staffLineDetection(img, staff_space, staff_height)
    show_images([removed_staff])
    horizontal_hist = np.sum(img, axis=1, keepdims=True)
    print(horizontal_hist)
