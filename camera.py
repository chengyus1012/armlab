"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
import math
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from rxarm import D2R
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError
from collections import OrderedDict
from utility_functions import Rx, ee_transformation_to_pose, Block
from scipy.spatial import distance

class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.BlockImageFrame = np.zeros_like(self.VideoFrame)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        self.DepthFrameHistory = np.zeros((5,720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([])
        self.extrinsic_matrix = np.array([])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_depths = np.array([])
        self.block_detections = np.array([])

        self.color_dict = OrderedDict({
            'red': (20, 20, 200),
            'orange': (0, 80, 250),
            'yellow': (0, 170, 255),
            'dark_green': (30, 110, 20),
            'light_green': (50, 135, 40),
            'blue': (110, 68, 10),
            'violet': (90, 70, 105)})

            # 'red': (0, 30, 180),
            # 'orange': (0, 90, 200),
            # 'yellow': (0, 175, 240),
            # # 'dark_green': (40, 100, 47),
            # 'light_green': (61, 130, 85),
            # 'blue': (100, 70, 53),
            # 'violet': (80, 60, 100)})

        self.rgb_colors = np.expand_dims(np.array(self.color_dict.values(), dtype='uint8'), 1)
        self.lab_colors = cv2.cvtColor(self.rgb_colors, cv2.COLOR_BGR2LAB)
        self.hsv_colors = cv2.cvtColor(self.rgb_colors, cv2.COLOR_BGR2HSV)

        self.mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        self.arm_mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        self.arm_base_mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)

        self.board_top = 110
        self.board_bottom = 700
        self.board_left = 190
        self.board_right = 1080
        self.arm_left = 560
        self.arm_right = 718
        self.arm_top = 374
        self.arm_bottom = 720
        self.latest_ee_pose = [0,10,10]
        self.depth_x_sample = 205

        # self.arm_base_mask = 
        self.arm_base_poly = np.array([[565,720],[565, 600],[540, 570],[540, 505], [595, 455], [675, 455], [730,505], [730,570], [700,600], [700,720]])
        self.arm_base_poly[:,0] += 15
        cv2.rectangle(self.mask, (self.board_left,self.board_top),(self.board_right,self.board_bottom), 255, cv2.FILLED)
        # cv2.rectangle(self.mask, (self.arm_left,self.arm_top),(self.arm_right,self.arm_bottom), 0, cv2.FILLED)
        cv2.fillPoly(self.mask, [self.arm_base_poly], 0)


    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        BlockImageFrame = self.VideoFrame.copy()
        blurred = cv2.GaussianBlur(BlockImageFrame, (5, 5), 0)
        lab_image = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for block, depth in zip(self.block_contours, self.block_depths):
            color_index, dist = self.retrieve_area_color(lab_image, block, self.lab_colors)
            min_color = self.color_dict.keys()[color_index]

            theta = cv2.minAreaRect(block)[2]
            M = cv2.moments(block)
            if M['m00'] == 0:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cv2.putText(BlockImageFrame, min_color + " " + str(int(depth)) , (cx-30, cy+40), font, 1.0, (255,255,0), thickness=2)
            cv2.putText(BlockImageFrame, str(int(theta)), (cx, cy), font, 1.0, (255,255,255), thickness=2)
            cv2.putText(BlockImageFrame, str(int(dist)), (cx+30, cy+40), font, 1.0, (255,50,50), thickness=2)

        cv2.rectangle(BlockImageFrame, (self.board_left,self.board_top),(self.board_right,self.board_bottom), (255, 0, 0), 2)
        # cv2.rectangle(BlockImageFrame,(self.arm_left,self.arm_top),(self.arm_right,self.arm_bottom), (255, 0, 0), 2)
        cv2.polylines(BlockImageFrame, [self.arm_base_poly], True, (255,0,0), 2)

        cv2.drawContours(BlockImageFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

        # if len(self.block_contours) > 0:
        #     print("Min area:",min(self.block_contours,key=lambda cnt: cv2.contourArea(cnt)),min([cv2.contourArea(cnt) for cnt in self.block_contours]))

        self.BlockImageFrame = BlockImageFrame
        


    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtBlockImageFrame(self):
        """!
        @brief      Converts block image frame to format suitable for Qt

        @return     QImage
        """
        try:
            frame = cv2.resize(self.BlockImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None



    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def loadCameraExtrinsic(self, file):
        """!
        @brief      Load camera extrinsic matrix from file.

        @param      file  The file
        """

        self.extrinsic_matrix = np.loadtxt(file).reshape((4,4))
        # self.extrinsic_matrix[2,3] += 10
        # self.extrinsic_matrix[:3,:3] = np.matmul(Rx(D2R*-2),self.extrinsic_matrix[:3,:3])
        print('Loaded extrinsic matrix from file')
        print(self.extrinsic_matrix)
        print(ee_transformation_to_pose(self.extrinsic_matrix))

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

        """
        min_depth = 500.0
        max_depth = 958.0


        # rgb_image = self.VideoFrame.copy()
        depth_data = self.DepthFrameRaw.copy()

        # TODO Use extrinsics to calculate delta
        top_depth = np.median(depth_data[self.board_top][self.depth_x_sample-2:self.depth_x_sample+3])
        bottom_depth = np.median(depth_data[self.board_bottom][self.depth_x_sample-2:self.depth_x_sample+3])
        delta_depth = (float(top_depth) - bottom_depth) / float(self.board_top - self.board_bottom)

        for r in range(self.board_top,depth_data.shape[0]):
            depth_data[r,:] += -(delta_depth * (r - self.board_top)).astype(np.uint16)
        
        blur_depth = cv2.medianBlur(depth_data,5)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(blur_depth, -1, sharpen_kernel)

        thresh = cv2.bitwise_and(cv2.inRange(sharpen, min_depth, max_depth), self.mask)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = filter(lambda cnt: cv2.contourArea(cnt) < 5000, contours)
        contours = filter(lambda cnt: cv2.contourArea(cnt) > 200, contours)

        block_contours = []
        top_depths = []
        for contour in contours:
            top_depth, new_mask = self.retrieve_top_depth(depth_data, contour)
            _, new_contours, _ = cv2.findContours(new_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            block_contours.extend(new_contours)
            top_depths.extend([top_depth]*len(new_contours))

        result = list(filter(lambda item: cv2.contourArea(item[0]) > 200, zip(block_contours,top_depths)))
        # block_contours = filter(lambda cnt: cv2.contourArea(cnt) > 200, block_contours)
        if len(result) == 0:
            block_contours = []
            top_depths = []
        else:
            block_contours, top_depths = [list(t) for t in list(zip(*result))]

        for i in range(len(block_contours)):
            epsilon = 0.1*cv2.arcLength(block_contours[i],True)

            block_contours[i] = (cv2.approxPolyDP(block_contours[i],epsilon,True))
            
        self.block_contours = block_contours
        self.block_depths = top_depths

        return block_contours, top_depths

    def getBlockColors(self, block_contours):
        BlockImageFrame = self.VideoFrame.copy()

        blurred = cv2.GaussianBlur(BlockImageFrame, (5, 5), 0)
        lab_image = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
        block_colors = []
        color_dists = []
        for block in block_contours:
            color_index, dist = self.retrieve_area_color(lab_image, block, self.lab_colors)
            min_color = self.color_dict.keys()[color_index]

            block_colors.append(min_color)
            color_dists.append(dist)

        return block_colors, color_dists
       

    def detect_blocks(self, ee_pose):
        depth_block_contours, top_depths = self.detectBlocksInDepthImage()

        block_colors, color_dists = self.getBlockColors(depth_block_contours)

        K = self.intrinsic_matrix
        H_camera_to_world = self.extrinsic_matrix

        detected_blocks = []
        for (block_contour, top_depth, block_color, color_dist) in zip(depth_block_contours, top_depths,block_colors, color_dists):
            theta = math.radians(cv2.minAreaRect(block_contour)[2] % 90.0)
            M = cv2.moments(block_contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            z = top_depth # TODO May want to sample from depth frame instead of using top depth

            # print(block_color, z)

            block_position_camera = z * np.matmul(np.linalg.inv(K),np.array([cx, cy, 1]).T)
            block_position_world = np.matmul(H_camera_to_world, np.concatenate([block_position_camera,[1]]))[:3]

            is_large = M['m00'] > Block.LARGE_BLOCK_THRESHOLD

            bottom_height = block_position_world[2] - (Block.LARGE_MM if is_large else Block.SMALL_MM)
            # for num_blocks_beneath in range(0,4): # Minimum no blocks beneath to maximum 4 tall stack, 3 beneath
            #     for num_small_blocks in range(num_blocks_beneath + 1):
            #         num_large_blocks = num_blocks_beneath - num_small_blocks
            #         stack_height = Block.SMALL_MM * num_small_blocks + Block.LARGE_MM * num_large_blocks

            #     if (math.isclose(bottom_height,0,abs_tol=1.0)):
            #         pass

            block = Block(block_position_world, theta, is_large=is_large, ignore=False, color=block_color, color_dist=color_dist)
            detected_blocks.append(block)      

        return detected_blocks

    def saveImage(self):
        cv2.imwrite("data/rgb_image.png", cv2.cvtColor(self.VideoFrame, cv2.COLOR_RGB2BGR))
        cv2.imwrite("data/raw_depth.png", self.DepthFrameRaw)

    def retrieve_area_color(self, data, contour, known_colors):
        mask = np.zeros(data.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean = cv2.mean(data, mask=mask)[:3]
        min_dist = (np.inf, None)
        for (i, color) in enumerate(known_colors):
            # print("Color:",color[0])
            # d = np.linalg.norm(color[0] - np.array(mean))
            d = distance.euclidean(color[0], mean)
            if d < min_dist[0]:
                min_dist = (d, i)
        return min_dist[1], min_dist[0]

    def retrieve_top_depth(self, depth, contour):
        mask = np.zeros(depth.shape, dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)

        top_depth = np.percentile(depth[mask == 255], 30)
        # for i in range(10):
        #     print("Contour",contour[0],"Percentile:",i*10,np.percentile(depth[mask == 255], i*10))
        mask = (cv2.bitwise_and(mask, cv2.inRange(depth, top_depth - 5, top_depth + 5))) #.astype(np.uint8) * 255
        return top_depth, mask

    # def world_xyz_to_world_xyz

class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera
        self.i=0

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        if self.i % 10 == 0:
            self.camera.processVideoFrame()
        self.i += 1


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        #for detection in data.detections:
        #print(detection.id[0])
        #print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        return
        # self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera
        self.i = 0

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        # top_depth = np.median(cv_depth[self.camera.board_top][self.camera.depth_x_sample-2:self.camera.depth_x_sample+3])
        # bottom_depth = np.median(cv_depth[self.camera.board_bottom][self.camera.depth_x_sample-2:self.camera.depth_x_sample+3])
        # delta_depth = (float(top_depth) - bottom_depth) / float(self.camera.board_top - self.camera.board_bottom)

        # for r in range(self.camera.board_top,cv_depth.shape[0]):
        #     cv_depth[r,:] += -(delta_depth * (r - self.camera.board_top)).astype(np.uint16)

        self.camera.DepthFrameRaw = cv_depth
        self.camera.DepthFrameHistory[1:5] = self.camera.DepthFrameHistory[0:4].copy()
        self.camera.DepthFrameHistory[0] = cv_depth
        if self.i % 10 == 0:
            self.camera.detectBlocksInDepthImage()
        self.i += 1

        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            block_frame = self.camera.convertQtBlockImageFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, block_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
