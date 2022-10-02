"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError
from collections import OrderedDict
from utility_functions import ee_transformation_to_pose

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
        self.block_detections = np.array([])

        self.color_dict = OrderedDict({
            'red': (50, 25, 160),
            'orange': (0, 65, 150),
            'yellow': (30, 150, 200),
            'dark_green': (60, 70, 35),
            'light_green': (95, 134, 50),
            'blue': (110, 65, 0),
            'violet': (75, 50, 63),
            'pink': (80, 50, 150)})

        self.rgb_colors = np.expand_dims(np.array(self.color_dict.values(), dtype='uint8'), 1)
        self.lab_colors = cv2.cvtColor(self.rgb_colors, cv2.COLOR_BGR2LAB)
        self.hsv_colors = cv2.cvtColor(self.rgb_colors, cv2.COLOR_BGR2HSV)

        self.mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        self.board_top = 120
        self.board_bottom = 670
        self.board_left = 200
        self.board_right = 1070
        self.arm_left = 560
        self.arm_right = 718
        self.arm_top = 374
        self.arm_bottom = 720
        cv2.rectangle(self.mask, (self.board_left,self.board_top),(self.board_right,self.board_bottom), 255, cv2.FILLED)
        cv2.rectangle(self.mask, (self.arm_left,self.arm_top),(self.arm_right,self.arm_bottom), 0, cv2.FILLED)


    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        BlockImageFrame = self.VideoFrame.copy()
        blurred = cv2.GaussianBlur(BlockImageFrame, (5, 5), 0)
        lab_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for block in self.block_contours:
            color_index, dist = self.retrieve_area_color(lab_image, block, self.hsv_colors)
            min_color = self.color_dict.keys()[color_index]

            theta = cv2.minAreaRect(block)[2]
            M = cv2.moments(block)
            if M['m00'] == 0:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cv2.putText(BlockImageFrame, min_color + " " + str(int(dist)) , (cx-30, cy+40), font, 1.0, (255,255,0), thickness=2)
            cv2.putText(BlockImageFrame, str(int(theta)), (cx, cy), font, 1.0, (255,255,255), thickness=2)
            # cv2.putText(self.BlockImageFrame, str(int()), (cx+30, cy+40), font, 1.0, (0,255,255), thickness=2)

        cv2.rectangle(BlockImageFrame, (self.board_left,self.board_top),(self.board_right,self.board_bottom), (255, 0, 0), 2)
        cv2.rectangle(BlockImageFrame,(self.arm_left,self.arm_top),(self.arm_right,self.arm_bottom), (255, 0, 0), 2)

        cv2.drawContours(BlockImageFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

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
        max_depth = 950.0

        depth_x_sample = 375

        rgb_image = self.VideoFrame
        depth_data = self.DepthFrameRaw

        # TODO Use extrinsics to calculate delta
        top_depth = depth_data[self.board_top][depth_x_sample]
        bottom_depth = depth_data[self.board_bottom][depth_x_sample]
        delta_depth = (float(top_depth) - bottom_depth) / float(self.board_top - self.board_bottom)

        for r in range(self.board_top,depth_data.shape[0]):
            depth_data[r,:] += -(delta_depth * (r - self.board_top)).astype(np.uint16)
        
        blur_depth = cv2.medianBlur(depth_data,5)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(blur_depth, -1, sharpen_kernel)

        thresh = cv2.bitwise_and(cv2.inRange(sharpen, min_depth, max_depth), self.mask)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        block_contours = []
        for contour in contours:
            top_depth, new_mask = self.retrieve_top_depth(depth_data, contour)
            _, new_contours, _ = cv2.findContours(new_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            block_contours.extend(new_contours)

        block_contours = filter(lambda cnt: cv2.contourArea(cnt) < 5000, block_contours)
        block_contours = filter(lambda cnt: cv2.contourArea(cnt) > 100, block_contours)
        
        self.block_contours = block_contours



    def saveImage(self):
        cv2.imwrite("data/rgb_image.png", cv2.cvtColor(self.VideoFrame, cv2.COLOR_RGB2BGR))
        cv2.imwrite("data/raw_depth.png", self.DepthFrameRaw)

    def retrieve_area_color(self, data, contour, known_colors):
        mask = np.zeros(data.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean = cv2.mean(data, mask=mask)[:3]
        min_dist = (np.inf, None)
        for (i, color) in enumerate(known_colors):
            d = np.linalg.norm(color[0] - np.array(mean))
            if d < min_dist[0]:
                min_dist = (d, i)
        return min_dist[1], min_dist[0]

    def retrieve_top_depth(self, depth, contour):
        mask = np.zeros(depth.shape, dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)

        top_depth = np.percentile(depth[mask == 255], 20)
        cv2.inRange
        mask = (cv2.bitwise_and(mask, cv2.inRange(depth, top_depth - 4, top_depth + 4))) #.astype(np.uint8) * 255
        return top_depth, mask


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
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
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
