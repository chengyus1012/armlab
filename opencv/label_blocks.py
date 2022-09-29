
#!/usr/bin/python
""" Example: 

python label_blocks.py -i image_blocks.png -d depth_blocks.png -l 905 -u 973

"""
from __future__ import print_function
import argparse
import sys
import cv2
import numpy as np

max_depth = 1050.0

top_board = 100
bottom_board = 666
depth_x_sample = 375

font = cv2.FONT_HERSHEY_SIMPLEX
colors = list((
    {'id': 'red', 'color': (10, 10, 127)},
    {'id': 'orange', 'color': (30, 75, 150)},
    {'id': 'yellow', 'color': (30, 150, 200)},
    {'id': 'green', 'color': (20, 60, 20)},
    {'id': 'blue', 'color': (100, 50, 0)},
    {'id': 'violet', 'color': (100, 40, 80)})
)

def retrieve_area_color(data, contour, labels):
    mask = np.zeros(data.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean = cv2.mean(data, mask=mask)[:3]
    min_dist = (np.inf, None)
    for label in labels:
        d = np.linalg.norm(label["color"] - np.array(mean))
        if d < min_dist[0]:
            min_dist = (d, label["id"])
    return min_dist[1] 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-d", "--depth", required = True, help = "Path to the depth image")
ap.add_argument("-l", "--lower", required = True, help = "lower depth value for threshold")
ap.add_argument("-u", "--upper", required = True, help = "upper depth value for threshold")
args = vars(ap.parse_args())
lower = int(args["lower"])
upper = int(args["upper"])
rgb_image = cv2.imread(args["image"])
cnt_image = cv2.imread(args["image"])
depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
top_depth = depth_data[top_board][depth_x_sample]
bottom_depth = depth_data[bottom_board][depth_x_sample]
delta_depth = (top_depth - bottom_depth) / float(top_board - bottom_board)
for r in range(top_board,depth_data.shape[0]):
    depth_data[r,:] += (delta_depth * (r - top_board)).astype(np.uint16)
# print(np.unique(depth_data))
# depth_data = (np.clip((depth_data)/max_depth,0,1)*255).astype(np.uint8)
# print(np.unique(depth_data))
cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
#cv2.namedWindow("Threshold window", cv2.WINDOW_NORMAL)
"""mask out arm & outside board"""
mask = np.zeros_like(depth_data, dtype=np.uint8)
cv2.rectangle(mask, (275,120),(1100,720), 255, cv2.FILLED)
cv2.rectangle(mask, (575,414),(723,720), 0, cv2.FILLED)
cv2.rectangle(cnt_image, (275,120),(1100,720), (255, 0, 0), 2)
cv2.rectangle(cnt_image, (575,414),(723,720), (255, 0, 0), 2)
depth_blur = cv2.GaussianBlur(depth_data, (3,3), 0)
print(depth_data.dtype, mask.dtype)

# thresh = cv2.bitwise_and(depth_data, mask)
# thresh = cv2.adaptiveThreshold(depth_data,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
# cv2.imshow('Thresh',thresh)
# thresh = cv2.bitwise_and(depth_blur, mask)
# cv2.

# edges = cv2.Canny(image=depth_data, threshold1=50, threshold2=100) # Canny Edge Detection
# print(np.unique(edges))
# Display Canny Edge Detection Image
# cv2.imshow('Canny Edge Detection', edges)
# depending on your version of OpenCV, the following line could be:
sobel_64 = cv2.Sobel(src=depth_data, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
abs_64 = np.absolute(sobel_64)
sobel_8u = np.uint16(abs_64)
thresh = cv2.inRange(sobel_8u, 5, 40)

print(np.unique(sobel_8u))
cv2.imshow('Canny Edge Detection', thresh)
img, contours, _ = cv2.findContours(depth_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=1)
# _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=1)
# for contour in contours:
#     color = retrieve_area_color(rgb_image, contour, colors)
#     theta = cv2.minAreaRect(contour)[2]
#     M = cv2.moments(contour)
#     cx = int(M['m10']/M['m00'])
#     cy = int(M['m01']/M['m00'])
#     cv2.putText(cnt_image, color, (cx-30, cy+40), font, 1.0, (0,0,0), thickness=2)
#     cv2.putText(cnt_image, str(int(theta)), (cx, cy), font, 0.5, (255,255,255), thickness=2)
#     print(color, int(theta), cx, cy)
# #cv2.imshow("Threshold window", thresh)
cv2.imshow("Image window", cnt_image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
