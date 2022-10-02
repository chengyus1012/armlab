
#!/usr/bin/python
""" Example: 

python label_blocks.py -i image_blocks.png -d depth_blocks.png -l 905 -u 973

"""
from __future__ import print_function
import argparse
import sys
import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

min_depth = 900.0
max_depth = 1000.0

top_board = 120
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

color_dict = OrderedDict({
    'red': (50, 25, 150),
    'orange': (30, 75, 150),
    'yellow': (30, 150, 200),
    'green': (20, 60, 20),
    'blue': (100, 50, 0),
    'violet': (100, 40, 80),
    'pink': (80, 50, 150)})


# lab_colors = np.zeros((len(colors), 1, 3), dtype="uint8")
rgb_colors = np.expand_dims(np.array(color_dict.values(), dtype='uint8'), 1)
lab_colors = cv2.cvtColor(rgb_colors, cv2.COLOR_BGR2LAB)
hsv_colors = cv2.cvtColor(rgb_colors, cv2.COLOR_BGR2HSV)
print(lab_colors)

def retrieve_area_color_lab(data, contour, known_colors):
    mask = np.zeros(data.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean = cv2.mean(data, mask=mask)[:3]
    min_dist = (np.inf, None)
    for (i, color) in enumerate(known_colors):
        d = np.linalg.norm(color[0] - np.array(mean))
        if d < min_dist[0]:
            min_dist = (d, i)
    return min_dist[1]

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


def retrieve_top_depth(depth, contour):
    mask = np.zeros(depth.shape, dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)

    top_depth = np.percentile(depth[mask == 255], 20)
    cv2.inRange
    mask = (cv2.bitwise_and(mask, cv2.inRange(depth, top_depth - 4, top_depth + 4))) #.astype(np.uint8) * 255
    return top_depth, mask

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
top_depth = depth_data[top_board+10][depth_x_sample]
bottom_depth = depth_data[bottom_board][depth_x_sample]
print(top_depth, bottom_depth)
delta_depth = (float(top_depth) - bottom_depth) / float(top_board - bottom_board)
print(top_depth, bottom_depth, top_board, bottom_board, delta_depth, top_depth - bottom_depth, float(top_board - bottom_board))

for r in range(top_board,depth_data.shape[0]):
    depth_data[r,:] += -(delta_depth * (r - top_board)).astype(np.uint16)
# for r in range(top_board,depth_data.shape[0]):
#     depth_data[r,:] += (delta_depth * (r - top_board)).astype(np.uint16)
# print(np.unique(depth_data))
# depth_data = (np.clip((depth_data)/max_depth,0,1)*255).astype(np.uint8)
# print(np.unique(depth_data))
cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
#cv2.namedWindow("Threshold window", cv2.WINDOW_NORMAL)
"""mask out arm & outside board"""
mask = np.zeros_like(depth_data, dtype=np.uint8)
cv2.rectangle(mask, (200,120),(1070,670), 255, cv2.FILLED)
cv2.rectangle(mask, (560,374),(718,720), 0, cv2.FILLED)
cv2.rectangle(cnt_image, (200,120),(1070,670), (255, 0, 0), 2)
cv2.rectangle(cnt_image,(560,374),(718,720), (255, 0, 0), 2)
rescaled_depth = ((1 -np.clip((depth_data - min_depth)/(max_depth-min_depth),0,1))*255).astype(np.uint8)

def mouse_callback(event, x, y, flags, params):
    if event == 2:
        print("coords", x, y, " colors Depth-" ,rescaled_depth[y,x], depth_data[y,x])

print(rescaled_depth.dtype)

# rescaled_depth *= mask
print(rescaled_depth.dtype)
# rescaled_depth = cv2.GaussianBlur(rescaled_depth, (7,7), 0)
rescaled_depth = cv2.medianBlur(rescaled_depth,5)
blur_depth = cv2.medianBlur(depth_data,5)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur_depth, -1, sharpen_kernel)

cv2.namedWindow('Rescaled')
depth_color = cv2.cvtColor(rescaled_depth, cv2.COLOR_GRAY2BGR)
cv2.waitKey(0)
print(depth_data.dtype, mask.dtype)

# thresh = cv2.bitwise_and(depth_data, mask)
thresh = cv2.bitwise_and(cv2.inRange(sharpen, lower, upper), mask)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# rescaled_depth = rescaled_depth[275:1100][120:720]
adapt_thresh = cv2.adaptiveThreshold(rescaled_depth, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -10)
# thresh = cv2.bitwise_and(thresh, mask)
# thresh *= mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# thresh = cv2.erode(thresh, kernel, 100)
# thresh = cv2.dilate(thresh, kernel, 50)
adapt_thresh = cv2.morphologyEx(adapt_thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow('Thresh',thresh)
cv2.imshow('Adaptive', adapt_thresh)
# cv2.



# edges = cv2.Canny(image=depth_data, threshold1=50, threshold2=100) # Canny Edge Detection
# print(np.unique(edges))
# Display Canny Edge Detection Image
# cv2.imshow('Canny Edge Detection', edges)
# depending on your version of OpenCV, the following line could be:
sobel_64x = cv2.Sobel(src=depth_data, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
abs_64x = np.absolute(sobel_64x)
sobel_64y = cv2.Sobel(src=depth_data, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
abs_64y = np.absolute(sobel_64y)
# sobel_8u = np.uint16(abs_64)
edges = cv2.bitwise_or(cv2.inRange(abs_64x, 15, 100),cv2.inRange(abs_64y, 15, 100))
edges = cv2.bitwise_and(edges,mask)
# print(np.unique(sobel_8u))
cv2.imshow('Canny Edge Detection', edges)
_, adapt_contours, _ = cv2.findContours(adapt_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=1)
_, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = filter(lambda cnt: cv2.contourArea(cnt) < 5000, contours)
contours = filter(lambda cnt: cv2.contourArea(cnt) > 30, contours)

print(len(contours))
# print(len(contours),contours)
cv2.drawContours(cnt_image, contours, -1, (0,0,255), thickness=2)
cv2.drawContours(depth_color, contours, -1, (0,0,255), thickness=2)

cv2.drawContours(cnt_image, adapt_contours, -1, (127,127,127), thickness=1)
cv2.drawContours(depth_color, adapt_contours, -1, (127,127,127), thickness=1)

cv2.drawMarker(cnt_image,(depth_x_sample,top_board),(255,255,0))
cv2.drawMarker(cnt_image,(depth_x_sample,bottom_board),(255,255,0))

cv2.imshow('Rescaled',depth_color)
cv2.setMouseCallback("Rescaled", mouse_callback)

blurred = cv2.GaussianBlur(rgb_image, (5, 5), 0)
lab_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


for contour in contours:
    top_depth, new_mask = retrieve_top_depth(depth_data, contour)
    cv2.imshow('Mask',new_mask)
    cv2.waitKey(0)
    # white = np.ones_like(cnt_image) * 255
    # cnt_image[new_mask == 255] += 30 

    _, new_contour, _ = cv2.findContours(new_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(cnt_image, new_contour, -1, (255,255,255), thickness=1)

    new_contour = new_contour[0]
    rgb_color = retrieve_area_color(rgb_image, contour, colors)
    color_index = retrieve_area_color_lab(lab_image, new_contour, lab_colors)
    lab_color = color_dict.keys()[color_index]

    color_index_hsv = retrieve_area_color_lab(hsv_image, new_contour, hsv_colors)
    hsv_color = color_dict.keys()[color_index]

    theta = cv2.minAreaRect(new_contour)[2]
    M = cv2.moments(new_contour)
    if M['m00'] == 0:
        continue
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.putText(cnt_image, lab_color , (cx-30, cy+40), font, 1.0, (255,255,0), thickness=2)
    cv2.putText(cnt_image, str(int(theta)), (cx, cy), font, 1.0, (255,255,255), thickness=2)
    cv2.putText(cnt_image, str(int(top_depth)), (cx+30, cy+40), font, 1.0, (0,255,255), thickness=2)

    print(rgb_color, lab_color, int(theta), cx, cy, top_depth)
# #cv2.imshow("Threshold window", thresh)
cv2.imshow("Image window", cnt_image)
while True:
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
