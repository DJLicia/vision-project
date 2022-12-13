import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector


# 0  is webcam ID number
vidcap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

buffer = 20
imgSize = 300

# for saving training data
folder = "Data/D"
numImages = 0

while True:
    success, img = vidcap.read()
    hands, img = detector.findHands(img)
    
    # crop hand image
    if hands:
        # for the first and only hand on screen
        hand = hands[0]
        x, y, w, h = hand['bbox']

        whiteImg = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        cropImg = img[y - buffer:y + h + buffer, x - buffer:x + w + buffer]

        # height, width, channel
        cropShape = cropImg.shape

        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize / h
            calcWidth = math.ceil(k * w)
            resizeImg = cv2.resize(cropImg, (calcWidth, imgSize))
            resizeShape = resizeImg.shape

            # adjust to center the cam image
            widthGap = math.ceil((300 - calcWidth) / 2)
            # overlays cam image ontop of whitebox
            whiteImg[:, widthGap:calcWidth + widthGap] = resizeImg
        
        else:
            k = imgSize / w
            calcHeight = math.ceil(k * h)
            resizeImg = cv2.resize(cropImg, (imgSize, calcHeight))
            resizeShape = resizeImg.shape

            # adjust to center the cam image
            heightGap = math.ceil((imgSize - calcHeight) / 2)
            whiteImg[heightGap:calcHeight + heightGap] = resizeImg


        cv2.imshow("CropImage", cropImg)
        cv2.imshow("WhiteImage", whiteImg)
    
    cv2.imshow("Image:", img)
    # 1 millisecond delay
    key = cv2.waitKey(1) 
    
    if key == ord("k"):
        numImages += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', whiteImg)
        print(numImages)