import os
import cv2

filelist = [file for file in os.listdir('.') if file.endswith('.png') and file.startswith('image_')] 
print(filelist)
images = [cv2.imread(f) for f in filelist]


bg_sub = cv2.createBackgroundSubtractorKNN()
bg_sub2 = cv2.createBackgroundSubtractorMOG2()

for i in range(10):
    for im in images:
        fg = bg_sub.apply(im)
        fg2 = bg_sub2.apply(im)
        cv2.imshow('KNN',fg)
        cv2.imshow('MOG', fg2)
        k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
