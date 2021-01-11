# New Char-Seg : returns a "Just The Char Image without neighbours" and its position in org image.

def char_seg2(org_img):
    img = np.copy(org_img)
    
    toshow = [img]
    
    labels = sk.measure.label(img, connectivity=1)
    lbl_num = np.max(labels[:, :])
    
    bounds = np.zeros((lbl_num+1, 4)) # [up, down, left, right]
    bounds[:, 0] = 99999999
    bounds[:, 2] = 99999999

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j]:
                bounds[labels[ i, j ]][0] = int(min(bounds[labels[ i, j ]][0], i))
                bounds[labels[ i, j ]][1] = int(max(bounds[labels[ i, j ]][1], i))
                bounds[labels[ i, j ]][2] = int(min(bounds[labels[ i, j ]][2], j))
                bounds[labels[ i, j ]][3] = int(max(bounds[labels[ i, j ]][3], j))

    only_char_arr = []
    
    for i in range(bounds.shape[0]):
        if bounds[i][0] == 99999999:
            continue
        cur = np.copy(labels[int(bounds[i][0]):int(bounds[i][1]+1), int(bounds[i][2]):int(bounds[i][3]+1)])
        cur = cur == i
        only_char_arr.append(cur)
            
    return [bounds, only_char_arr]
