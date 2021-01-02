from commonfunctions import *
import skimage as sk
import numpy as np
import matplotlib as mp
import scipy as sp
import heapq
from PIL import Image


def binarize(img, block_size=35):
    t = sk.filters.threshold_local(img, block_size, offset=10)
    img_b = img < t
    return img_b


def deskew(img, delta=1, limit=60):
    def find_score(arr, angle):
        data = sp.ndimage.interpolation.rotate(
            arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle: '+str(best_angle))
    # correct skew
    img_n = sp.ndimage.interpolation.rotate(
        img, best_angle, reshape=False, order=0)
    return img_n


def line_detection(img):

    bounding_boxes = sk.measure.find_contours(img, 0.8)
    newImg = np.zeros(img.shape)
    lines = []
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = [np.min(box[:, 1]), np.max(
            box[:, 1]), np.min(box[:, 0]), np.max(box[:, 0])]
        Ymin -= 2
        Ymax += 2
        ar = (Xmax-Xmin)/(Ymax-Ymin)
        if ((ar > 5)):
            rr, cc = sk.draw.rectangle(
                start=(Ymin, Xmin), end=(Ymax, Xmax), shape=newImg.shape)
            rr = rr.astype(int)
            cc = cc.astype(int)
            newImg[rr, cc] = True

            # lines.append(int(Ymin))
            # lines.append(int(Ymax))
            # print("sdsd")
    # show_images([newImg])
    #se = np.ones((1, 21))
    #newImg = sk.morphology.dilation(newImg, se)
    #newImg = sk.morphology.dilation(newImg, se)
    #newImg = sk.morphology.closing(newImg)
    # newImg = sk.morphology.closing(newImg)
    # newImg = sk.morphology.closing(newImg)
    bounding_boxes = sk.measure.find_contours(newImg, .1)
    newImg = np.zeros(img.shape)
    lines = []
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = [np.min(box[:, 1]), np.max(
            box[:, 1]), np.min(box[:, 0]), np.max(box[:, 0])]
        Ymin -= 2
        Ymax += 2
        ar = (Xmax-Xmin)/(Ymax-Ymin)
        if ((ar > 5)):
            rr, cc = sk.draw.rectangle(
                start=(Ymin, Xmin), end=(Ymax, Xmax), shape=newImg.shape)
            rr = rr.astype(int)
            cc = cc.astype(int)
            newImg[rr, cc] = True

            lines.append(int(Ymin))
            lines.append(int(Ymax))
    show_images([newImg])
    return lines


def convertToLineSpace(row, lines):
    if row < lines[0]:
        return (-1, -1)
    if row == lines[0]:
        return (0, 0)
    for i in range(1, len(lines)):
        if row == lines[i]:
            if i % 2 == 0:
                return (5*(i)//2, 5*(i)//2)
            else:
                return (5*((i)//2) + 4, 5*((i)//2) + 4)
        if row < lines[i] and row > lines[i-1]:
            x = (lines[i]-lines[i-1])/3
            y = int((row-lines[i-1])/x)
            if i % 2 == 1:
                return ((5*((i-1)//2))+y, (5*((i-1)//2))+y+1)
            else:
                return (5*((i-2)//2) + 4, 5*((i-2)//2) + 5)
    return (5*len(lines), 5*len(lines))


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


def modelStaffLineShape(staffImg, K=5, T=5):
    O_arr = []
    for i in range(staffImg.shape[1]):
        conn_comp = runs_of_ones_array(img[:, i])
        if len(conn_comp) < T:
            continue
        j = 0
        indx = 0
        O_I = 0

        while j < img.shape[0]:
            if img[j][i] == True:
                O_T = 0
                k = 1
                mid = int(j + conn_comp[indx]//2)
                while k < K and i+k < img.shape[1]:
                    c1 = runs_of_ones_array(img[:, i+k])
                    j1 = 0
                    indx1 = 0
                    dist = 10000
                    while j1 < img.shape[0]:
                        if img[j1][i+k] == True:
                            mid1 = int(j1+c1[indx1]//2)
                            if abs(dist) > abs(mid1-mid):
                                dist = mid1-mid
                            else:
                                break
                            j1 += c1[indx1]-1
                            indx1 += 1
                        j1 += 1

                    O_T += (1/(i+k))*(dist)
                    k += 1
                j += conn_comp[indx]-1
                indx += 1
                O_I += O_T
            j += 1
        O_I *= (1/len(conn_comp))
        O_arr.append(O_I)
    return O_arr


def removeStaffLines(img, T_LEN):
    newImg = np.copy(img)
    for i in range(0, img.shape[1]):
        arr = runs_of_ones_array(img[:, i])
        # print(arr)
        k = 0
        j = 0
        while j < img.shape[0]:
            if img[j][i] == True:
                if arr[k] < T_LEN:
                    for x in range(0, arr[k]):
                        newImg[j][i] = False
                        j += 1
                else:
                    j += arr[k]-1
                k += 1
            j += 1
    return newImg


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
    indx = 0
    #graph = []
    n = img.shape[0]*img.shape[1]
    graph = {}
    print(len(graph))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(4):
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
    # u = []
    p = []
    indxStart = r*img.shape[1]-p1
    n = len(graph)  # 2*img.shape[0]*img.shape[1]
    INF = 1000000000
    print("N "+str(n))
    for i in range(n):
        dis.append(INF)
        # u.append(False)
        p.append(-1)
    dis[indxStart] = 0
    p[indxStart] = indxStart
    priority_q = []
    finished = set()
    relaxEdges(indxStart, 0, dis, priority_q)
    # for i in range(n):
    #     v = -1
    #     for j in range(n):
    #         if u[j] == False and (v == -1 or dis[j] < dis[v]):
    #             v = j
    #     if dis[v] == INF:
    #         break
    #     u[v] = True
    #     for edge in graph[v]:
    #         if dis[v]+edge['w'] < dis[edge['node']]:
    #             dis[edge['node']] = dis[v]+edge['w']
    #             p[edge['node']] = v
    indx = 0
    while len(priority_q) and len(finished) != n and indx < n:
        _, u = heapq.heappop(priority_q)
        finished.add(u)
        indx += 1
        for edge in graph[u]:
            if edge['node']-p1 >= 0 and edge['node']-p1 < n and dis[edge['node']-p1] > dis[u]+edge['w']:
                relaxEdges(edge['node']-p1, dis[u]+edge['w'], dis, priority_q)
                p[edge['node']-p1] = u
    return p


def getPath(img, p, r, p1):
    indxStart = r*img.shape[1]-p1
    path = []
    v = indxStart+img.shape[1]-1
    while not(v == indxStart):
        # print(v)
        path.append(v)
        v = p[v]
    path.append(indxStart)
    return path


def getWhitePercent(img, path, p1):
    b = 0
    w = 0
    x = 0
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
    lastRun = 0
    startIndx = 0
    endIndx = 0
    for i in range(len(path)):
        path[i] += p1
        r = path[i]//img.shape[1]
        c = path[i] % img.shape[1]
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
    white_perc = 0.75
    graph = constructGraph(img)
    newImg = np.zeros(img.shape)
    strip_height = 2 * staff_space
    print("strip hegiht "+str(strip_height))
    for i in range(img.shape[0]):
        print("row "+str(i))
        p1 = max(0, i-strip_height)
        p2 = min(img.shape[0]-1, i+strip_height//2)
        print("P1 "+str(p1)+" P2 "+str(p2))
        # p2 = 1
        print(p1*img.shape[1], (p2+1) * img.shape[1])
        p = dijkstra(img, graph[p1*img.shape[1]: (p2+1)
                                * img.shape[1]], i, p1*img.shape[1], p2)
        # print(p)
        print("finished dijksra")
        path = getPath(img, p, i, p1*img.shape[1])
        print("finished get path")
        if getWhitePercent(img, path, p1*img.shape[1]) < white_perc:
            continue
        # staffLines.append(path)
        path = trimPath(img, path, strip_height, p1*img.shape[1])
        for i in path:
            r = i//img.shape[1]
            c = i % img.shape[1]
            newImg[r][c] = True
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


# show_images(newImgs)
#staff_height, staff_space = verticalRunLength(img)

#T_LEN = min(2*staff_height, staff_height+staff_space)


# 1130,2261
# print(img.shape)
#show_images([img, staffLineDetection(img, staff_space, staff_height)])
# show_images([img])

img = sk.io.imread('imgs/test10.jpg', as_gray=True)
img = sk.transform.resize(img, (400, 400))
show_images([img])
img = img.astype(np.float64) / np.max(img)
img = 255 * img
img = img.astype(np.uint8)
img = binarize(img)
img = deskewImg(img)
#img = img > 0
show_images([img])
staff_height, staff_space = verticalRunLength(img)
