def deskew(original_img):
    img = np.copy(rgb2gray(original_img))
    
    # Canny
    imgCanny = canny(img, sigma=1.5)
    thresh = threshold_otsu(imgCanny)
    imgCanny = (imgCanny >= thresh)
    
    # Apply Hough Transform
    angleSet = np.linspace(-np.pi, np.pi, 1440)  # Generates a list of 360 Radian degrees (-pi/2, pi/2)
    houghArr, theta, dis = hough_line(imgCanny, angleSet)

    flatIdx = np.argmax(houghArr)
    bestTheta = (flatIdx % theta.shape[0])
    bestTheta = angleSet[bestTheta]
    bestDis = np.int32(np.floor(flatIdx / theta.shape[0]))
    bestDis = dis[bestDis]
    

    # Rotate
    thetaRotateDeg = (bestTheta*180)/np.pi
    if thetaRotateDeg > 0 :
        thetaRotateDeg = thetaRotateDeg - 90
    else : 
        thetaRotateDeg = thetaRotateDeg + 90

    imgRotated = rgb2gray(transform.rotate(img, thetaRotateDeg, resize=True))
    return imgRotated
