import os
import cv2
import numpy as np
\
filelist = [file for file in os.listdir('.') if file.endswith('.png') and file.startswith('image_')] 
print(filelist)
images = [cv2.imread(f) for f in filelist]


bg_sub = cv2.createBackgroundSubtractorKNN()
bg_sub2 = cv2.createBackgroundSubtractorMOG2()

for i in range(10):
    for img in images:
        # fg = bg_sub.apply(im)
        # fg2 = bg_sub2.apply(im)
        # cv2.imshow('KNN',fg)
        # cv2.imshow('MOG', fg2)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,5,3,0.04)
        # dst = cv2.dilate(dst,None)
        img[dst>0.01*dst.max()]=[0,0,255]

        cv2.imshow('dst', img)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            exit()
