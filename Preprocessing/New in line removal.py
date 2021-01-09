def cher_seg(img_o, t):
    char = []
    
        
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
        if row_start_freq[i]/max_freq_row_start >= 0.1:
            row_starts.append(i)
    return row_starts


def without_lines2(img, T_LEN):
    staff_rows_starts = get_lines_rows(img, T_LEN)
    print(staff_rows_starts)
    
    newImg = np.zeros(img.shape)
    
    for i in range(0, img.shape[1]):
        arr = runs_of_ones_array(img[:, i])
        block_num = 0
        row = 0
        while row < img.shape[0]:
#             print("*************\nRow: {0}, Col: {1}".format(row, i))
            if img[row][i] == True:
                
                if row in staff_rows_starts or row-1 in staff_rows_starts:
#                     print("Staff Start, block size: {0}".format(arr[block_num]))
                    row += T_LEN 
                    arr[block_num] -= T_LEN
                    arr[block_num] = max(arr[block_num], 0)
                    
#                     print("TMP: ", arr[block_num])
                    if arr[block_num] > 0:
                        block_num -= 1
                
                else:
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
                    

############ mine ############
img = rgb2gray(io.imread('imgs/PublicTestCases/test-set-scanned/test-cases/10.PNG'))#[:, ]
# img = rgb2gray(io.imread('imgs/PublicTestCases/test-set-camera-captured/test-cases/24.jpg'))#[:800, :]


to_show = []

# Deskew
img = deskew(img)

# Binarization
img = img.astype(np.float64) / np.max(img)
img = 255 * img
img = img.astype(np.uint8)
img = binarize(img, 101)
img = np.array(img)

to_show.append(img)
print(type(img))

img = sk.morphology.binary_dilation(img, np.ones( (2, 1) )  )

to_show.append(img)

# Get Staff-Depth, Staffs-Space
staff_height, staff_space = verticalRunLength(img)
# T_LEN = min(2*staff_height, staff_height+staff_space)
T_LEN = staff_height

# Extract filled circles
img_1 = extractCircleNotes(img, staff_height)
img_1 = img_1 > 0

# to_show.append(img_1)

# Get Lines
lines = classicLineSegmentation(img, staff_space)
show(to_show)

# Img without lines.
removed_staff = without_lines2(img, T_LEN)
removed_staff = removed_staff > 0
removed_staff = removed_staff | img_1

# removed_staff = gaussian(removed_staff, sigma=1.2)

removed_staff = sk.morphology.binary_opening(removed_staff, np.ones((2,1)))
removed_staff = removed_staff == 1


# removed_staff = sk.morphology.binary_dilation(removed_staff_d, np.ones((2, 2)))

to_show.append(removed_staff)

show(to_show)

io.imsave('Char_Seg Out/Scanned/tmp-10-opn-me.jpg', removed_staff.astype(int))

