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


def extractMusicalNotes(img, T_LEN):
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


def extractStaffLines(img, T_LEN):
    newImg = np.copy(img)
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
                        newImg[j][i] = False
                        # if np.sum(img[j, 0:i])+np.sum(img[j, i:img.shape[1]]) > 0.5*img.shape[1]:
                        #     newImg[j][i] = True
                        j += 1
                else:
                    j += arr[k]-1
                k += 1
            j += 1
    return newImg


def extractStaffLines2(img, T_LEN, img_o):
    newImg = np.copy(img)
    cdef int k = 0
    cdef int i = 0
    cdef int j = 0
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
                            # print("dd")
                            pass
                        j += 1
                else:
                    j += arr[k]-1
                k += 1
            j += 1
    return newImg


def extractCircleNotes(img, staff_space):
    # print(staff_height)
    newImg = np.copy(img)
    # se = np.ones(((staff_height)*2, (staff_height)*2))
    # newImg = sk.morphology.binary_dilation(newImg, se)
    se = np.ones(((staff_space), (staff_space)))
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


def estimateStaffLines(img, staff_height):
    newImg = np.copy(img)
    # newImg = sk.morphology.skeletonize(img)
    # se = np.ones((1, 15))
    # newImg = sk.morphology.binary_closing(newImg, se)
    # show_images([newImg])
    # detect lines
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = sk.transform.hough_line(newImg, theta=tested_angles)

    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(newImg, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]),
                         np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap=cm.gray, aspect=1/1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(newImg, cmap=cm.gray)
    origin = np.array((0, newImg.shape[1]))
    cdef int lines = 0
    cdef float maxDis = -1000000
    y0 = -1
    y1 = -1
    for _, angle, dist in zip(*sk.transform.hough_line_peaks(h, theta, d)):
        # print(dist)
        if dist > maxDis:
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            maxDis = dist
        lines += 1
    ax[2].plot(origin, (y0, y1), '-r')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((newImg.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected First Line')

    plt.tight_layout()
    plt.show()
    print("-------------\ndetected lines "+str(lines)+"\n----------------")


def getFlatHeadNotePos(staff_lines, note, staff_space, charPos, staff_height, img_o=None):
    img = np.copy(note)
    s_c = np.copy(staff_lines)
    n_c = np.copy(note)
    s_c = s_c > 0
    n_c = n_c > 0
    s_c[charPos[0]:charPos[1], charPos[2]:charPos[3]
        ] = s_c[charPos[0]:charPos[1], charPos[2]:charPos[3]] | n_c
    err = staff_space//8

    # fill notes
    # se = np.ones((staff_space, staff_space))
    edges = sk.feature.canny(note)
    # filled_note = sp.ndimage.binary_fill_holes(edges, se)
    # filled_note = sp.ndimage.binary_fill_holes(filled_note)
    labels = sk.morphology.label(edges)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    edges[labels != background] = True
    show_images([edges])
    ############
    # np.ones((staff_space, staff_space))
    se = sk.morphology.disk(staff_space//2)
    img = sk.morphology.binary_opening(img, se)
    img = sk.morphology.binary_erosion(img, se)
    img = sk.morphology.binary_erosion(img)
    se = sk.morphology.disk(staff_space//4)
    img = sk.morphology.binary_dilation(img, se)
    #staff_lines = sk.morphology.binary_opening(staff_lines)

    # show_images([staff_lines])
    bounding_boxes = sk.measure.find_contours(img, 0.8)
    # newImg = np.zeros(img.shape)
    # print(len(bounding_boxes))
    output = []
    for box in bounding_boxes:
        # print(np.max(box[:,1]))
        # box = np.uint8(box)
        # print(box)
        [Xmin, Xmax, Ymin, Ymax] = [np.min(box[:, 1]), np.max(
            box[:, 1]), np.min(box[:, 0]), np.max(box[:, 0])]
        ar = (Xmax-Xmin)/(Ymax-Ymin)
        if ar >= 0.5 and ar <= 1.5:
            r0 = int(Ymin)  # -staff_height*2
            r1 = int(Ymax)  # +staff_height*2
            c0 = int(Xmin)  # -staff_space
            c1 = int(Xmax)  # +staff_space
            r0 = max(r0, 0)
            r1 = min(r1, staff_lines.shape[0])
            c0 = max(c0, 0)
            c1 = min(c1, staff_lines.shape[1])
            center = (r0+r1)//2
            center2 = staff_lines.shape[1]//2
            #print("row0 "+str(r0) + " row1 " + str(r1))
            horz_hist = np.sum(
                staff_lines[center-staff_height:center+staff_height, :], axis=1)
            # print(horz_hist)
            maximum = np.sum(horz_hist)

            # up = np.sum(staff_lines[0:center-staff_height//2, center2])
            # down = np.sum(staff_lines[center+staff_height//2:, center2])
            # print(staff_lines[:, staff_lines.shape[1]//2])
            # print("UP")
            up_arr = runs_of_ones_array(
                staff_lines[0:center-staff_height//2, center2])
            up = 0
            t = staff_height//2
            for x in up_arr:
                if x >= t:
                    up += 1
            # print(runs_of_ones_array(
            #    staff_lines[0:center-staff_height//2, center2]))
            # print("DOWN")
            # print(down)
            #print("Staff Height")
           # print(staff_height)
            # err = staff_height//2
            if maximum < staff_lines.shape[1]:
                # print("space")
                if up == 0:
                    x = 0
                    i = center
                    while i > 0 and staff_lines[i, center2] == False:
                        x += 1
                        i += 1
                    if x < staff_space:
                        # print("g2")
                        output.append("g2")
                    elif x < 2*staff_space:
                        # print("a2")
                        output.append("a2")
                    else:
                        # print("b2")
                        output.append("b2")
                elif up == 1:
                    # print("e2")
                    output.append("e2")
                elif up == 2:
                    # print("c2")
                    output.append("c2")
                elif up == 3:
                    # print("a")
                    output.append("a")
                elif up == 4:
                    # print("f")
                    output.append("f")
                else:
                    x = 0
                    i = center
                    while i < staff_lines.shape[0] and staff_lines[i, center2] == False:
                        x += 1
                        i -= 1
                    if abs(x) < staff_space:
                        # print("d")
                        output.append("d")
                    else:
                        # print("c")
                        output.append("c")
                    # print("c or d")
            else:
                # print("line")
                if up == 5:
                    x = 0
                    i = center
                    while i < staff_lines.shape[0] and staff_lines[i, center2] == False:
                        x += 1
                        i -= 1
                    if abs(x) < staff_space:
                        # print("d")
                        output.append("d")
                    else:
                        # print("c")
                        output.append("c")
                elif up == 1:
                    # print("d2")
                    output.append("d2")
                elif up == 2:
                    # print("b")
                    output.append("b")
                elif up == 3:
                    # print("g")
                    output.append("g")
                elif up == 4:
                    # print("e")
                    output.append("e")
                elif up == 0:
                    x = 0
                    i = center
                    while i > 0 and staff_lines[i, center2] == False:
                        x += 1
                        i += 1
                    if x < staff_space:
                        # print("g2")
                        output.append("g2")
                    elif x < 2*staff_space:
                        # print("a2")
                        output.append("a2")
                    else:
                        # print("b2")
                        output.append("b2")

            # show_images([s_c[charPos[0]:charPos[1],
            #                 charPos[2]: charPos[3]], s_c, note])
        # ar = 1/ar
        # if ( ( ar<=3.5 and ar>= 2.5)):

        # print(ar)
        # rr, cc = sk.draw.rectangle(start=(Ymin, Xmin), end=(
        #     Ymax, Xmax), shape=newImg.shape)

        # rr = rr.astype(int)
        # cc = cc.astype(int)
        # newImg[rr, cc] = True

    return output


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
            # print(patch.shape)
            for k in range(patch.shape[0]):
                x = np.sum(patch[k, :])
                if x >= 0.2*patch.shape[1]:
                    patch[k, :] = img_o[i*patch_height: (
                        i+1)*patch_height, j*patch_width: (j+1)*patch_width][k, :]
    return img


def runCode():
    cdef int T_LEN
    folder = 'test_new'
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
        img_o = np.copy(img)
        # show_images([img_o])
        img = sk.morphology.binary_dilation(img)
        staff_height, staff_space = verticalRunLength(img)
        T_LEN = min(2*staff_height, staff_height+staff_space)
        ########
        staffLines = extractStaffLines(img, T_LEN)
        se = np.ones((1, 55))
        # #staffLines = sk.morphology.thin(staffLines)
        # staffLines = sk.morphology.binary_opening(staffLines, se)
        # staffLines = sk.morphology.binary_dilation(staffLines, se)
        # se = np.ones((1, 150))
        # staffLines = sk.morphology.binary_dilation(staffLines, se)
        # show_images([(staffLines)])
        fixed_staff_lines = extractStaffLines2(staffLines, T_LEN, img)
        # show_images([(fixed_staff_lines)])
        fixed_staff_lines = sk.morphology.binary_opening(fixed_staff_lines, se)
        # show_images([(fixed_staff_lines)])
        fixed_staff_lines = fixStaffLines(
            fixed_staff_lines, staff_height, staff_space, img_o)
        # show_images([(fixed_staff_lines)])
        fixed_staff_lines = sk.morphology.binary_opening(fixed_staff_lines, se)
        # show_images([(fixed_staff_lines)])
        # io.imsave(folder + "Out/"+filename+"fixed_staff.png",
        #          sk.img_as_uint(fixed_staff_lines))
        # continue
        # show_images([fixed_staff_lines])
        #########
        ########
        print(staff_space)
        img_1 = extractCircleNotes(img, staff_space)
        img_1 = img_1 > 0
        # show_images([img_1])
        ##########
        #####
        lines = classicLineSegmentation(img, staff_space)
        ######
        print(img.shape)
        # removed_staff, staff_lines = staffLineDetection(img, staff_space, staff_height)
        # show_images([removed_staff])
        removed_staff = extractMusicalNotes(img, T_LEN)
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
                # estimateStaffLines(
                #     staffLines[line[0]:line[1], line[2]:line[3]], staff_height)
                # continue
                i += 1
                chars = charachterSegmentation(
                    removed_staff[line[0]: line[1], line[2]: line[3]], 0*staff_height*2)
                try:
                    os.mkdir(folder + "Out/"+filename +
                             "_img/"+filename+"_lines"+str(i))
                except Exception as e:
                    pass
                j = 0
                lineOut = []
                for char in chars:
                    # print(char)
                    # show_images(
                    #     [removed_staff[line[0]:line[1], line[2]:line[3]][char[0]:char[1], char[2]:char[3]]])
                    out = getFlatHeadNotePos(fixed_staff_lines[line[0]: line[1], line[2]: line[3]], removed_staff[line[0]: line[1],
                                                                                                                  line[2]: line[3]][char[0]: char[1], char[2]: char[3]], staff_space, char, staff_height)
                    lineOut.append(
                        out)
                    io.imsave(folder + "Out/"+filename+"_img/"+filename+"_lines"+str(i)+"/"+filename+"_line_"+str(i) + "_char_"+str(
                        j)+".png", sk.img_as_uint(removed_staff[line[0]: line[1], line[2]: line[3]][char[0]: char[1], char[2]: char[3]]))

                    j += 1
                print("--------Line----------")
                print(lineOut)
            except Exception as e:
                print(e)
                continue
        # io.imsave(folder+"Out/"+filename+"_removed_stadd_closing.png",sk.img_as_uint(removed_staff))
        # show_images([removed_staff])
        # staffs = lineSegmentation(removed_staff, staff_lines)

        # chars = charachterSegmentation(staffs[0], 0)
        print("done "+filename)
        # io.imsave("output/test7_removed_stadd_closing.png",removed_staff)
        # show_images(chars)