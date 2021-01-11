# Decide if should be rotate by 180deg, and Do it.

def count_matchings(char):
    # Cnt is the #matching with the dataset
    cnt = 0
    # Arr is the list of dataset files
    arr = os.listdir('skel/perf')
    # Convert the input char to be a 3D list and convert boolean values to int
    x = np.zeros((char.shape[0], char.shape[1], 3))
    char = np.multiply(char, 1)
    for i in range(3):
        x[:, :, i] = char

    # Resizing the input char to be (400x400)
    x = sk.transform.resize(x, (400, 400))
    x = cv2.cvtColor(255 * x.astype("uint8"), cv2.COLOR_BGR2RGB)
    
#     show([x], ["Input Char"])
    
    # Count matchings between images in dataset and the input char
    for y in arr:
        z = cv2.imread('skel/perf/' + y)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        z = cv2.resize(z, (400, 400))
        
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(x, None)
        kp2, des2 = sift.detectAndCompute(z, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = []
        if len(kp) and len(kp2):
            matches = bf.match(des,des2)
            matches = sorted(matches, key = lambda x:x.distance)

        cnt += len(matches)
    
#     print("#Matching : {0}".format(cnt))
    # Return the number of matchings
    return cnt

def decide_if_rotate180(img, chars_bounds):
    best_index = 0
    best_count = 0

    for i in range(len(chars_bounds)):
        char_pos = np.array(chars_bounds[i]).astype(int)
        if char_pos[0] > img.shape[0]:
            continue
        char = img[ char_pos[0]:char_pos[1]+1, char_pos[2]:char_pos[3]+1 ]
        cur = count_matchings(np.copy(char))
        
        if cur > best_count:
            best_count = cur
            best_index = i
    
#     clef = img[ chars_bounds[best_index][0]:chars_bounds[best_index][1]+1, chars_bounds[best_index][2]:chars_bounds[best_index][3]+1 ]
#     print("########################################\nMost Match is:")
#     show([clef])
#     print("########################################")
    
    # If the clef's right border is in the right part of the image, rotate it image by 180 deg.
    if chars_bounds[ best_index ][3] > img.shape[1]/2:
        img = sk.transform.rotate(img, 180, resize=False, mode='constant', cval=1)
        
    return img