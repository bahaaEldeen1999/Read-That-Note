def binarize(img, block_size=35):
    t = sk.filters.threshold_local(img, block_size, offset=10)
#     t = sk.filters.threshold_otsu(img)
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
    
#     print('Max Freq : ', max_freq_row_start)
    
    for i in range(len(row_start_freq)):
        # Approximately, if the row "i" is frequently treated as a starting of staffs with this ratio
        # by the most frequnt starting row, then consider it as a starting row of staffs.
        if row_start_freq[i] >= img.shape[1]/20:
            row_starts.append(i)
    return [row_starts, row_start_freq]


def without_lines2(img, T_LEN):
    staff_rows_starts, row_start_freq = get_lines_rows(img, T_LEN)
    is_here = np.zeros((1, img.shape[0] + 10))[0]
    for x in staff_rows_starts:
        is_here[x] = 1
    
    print(staff_rows_starts)
    
    newImg = np.zeros(img.shape)
    
#     for r in staff_rows_starts:
#         newImg[r, :] = True
    
    for i in range(0, img.shape[1]):
        arr = runs_of_ones_array(img[:, i])

        block_num = 0
        row = 0
        while row < img.shape[0]:
            if img[row][i] == True:
                
                found = False
                for idx in range(-int(T_LEN), int(T_LEN)):
                    if row - idx >= 0 and row - idx < img.shape[0]:
                        # If near to a good line, and about to be a line
                        if is_here[row - idx] and row_start_freq[row] >= img.shape[1]/25:
                            found = True
                            jump = T_LEN
                            row += jump 
                            arr[block_num] -= jump

                            arr[block_num] = max(arr[block_num], 0)

                            if arr[block_num] > 0:
                                block_num -= 1
                            break
                
                if found == False:
#                     print("To Fill: {0}".format(arr[block_num]))
                    for item in range(arr[block_num]):
                        if row >= img.shape[0]:
                            break
                        newImg[row][i] = True
                        row += 1
                        
                row -= 1
                block_num += 1
            row += 1
    return newImg
           


def extractCircleNotes(img, staff_height):
    newImg = np.copy(img)
    #################  We May Littilize Factors of the SE  #################
    se = np.ones(((staff_height+1)*2, (staff_height+1)*2))
    newImg = sk.morphology.binary_opening(newImg, se)
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


def deskew(original_img):
    img = np.copy((original_img))
    # Canny
    imgCanny = sk.feature.canny(img, sigma=1.5)
    thresh = sk.filters.threshold_otsu(imgCanny)
    imgCanny = (imgCanny >= thresh)

    # Apply Hough Transform
    angleSet = np.linspace(-np.pi, np.pi, 2880)
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

    imgRotated = (sk.transform.rotate(img, thetaRotateDeg, resize=True, mode='constant', cval=1))
    return imgRotated


def runCode():
    folder = 'imgs/PublicTestCases/test-set-scanned/test-cases'
    try:
        os.mkdir(folder + "Out")
    except:
        pass
    for filename in os.listdir(folder):
        print("file "+str(filename))
        img = sk.io.imread(os.path.join(folder, filename), as_gray=True)

        img = deskew(img)

        img = img.astype(np.float64) / np.max(img)
        img = 255 * img
        img = img.astype(np.uint8)
        img = binarize(img, 101)
        
        img = sk.morphology.binary_dilation(img)
        
        staff_height, staff_space = verticalRunLength(img)
        T_LEN = staff_height#min(2*staff_height, staff_height+staff_space)
        ########
        img_1 = extractCircleNotes(img, staff_height)
        img_1 = img_1 > 0
        ##########
        #####
        lines = classicLineSegmentation(img, staff_space)
        ######
        print(img.shape)

        removed_staff = without_lines2(img, T_LEN)
        removed_staff = removed_staff > 0
        removed_staff = removed_staff | img_1
        
        se = np.ones((staff_height*2+1, staff_height*2+1))
        removed_staff_d = np.copy(removed_staff)
        removed_staff_d = sk.morphology.binary_dilation(removed_staff_d, se)

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

                    io.imsave(folder + "Out/"+filename+"_img/"+filename+"_lines"+str(i)+"/"+filename+"_line_"+str(i) + "_char_"+str(
                        j)+".png", sk.img_as_uint(removed_staff[line[0]:line[1], line[2]:line[3]][char[0]:char[1], char[2]:char[3]]))
                    j += 1
            except:
                continue

        print("done "+ filename)


runCode()