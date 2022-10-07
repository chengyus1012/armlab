#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
from http.client import TEMPORARY_REDIRECT
import os

from utility_functions import transformation_from_world_to_arm
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import sys
import cv2
import numpy as np
import rospy
import time
from functools import partial

from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer, QCoreApplication)
from PyQt4.QtGui import (QPixmap, QImage, QApplication, QWidget, QLabel,
                         QMainWindow, QCursor, QFileDialog)

from ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread
from kinematics import *
from modern_robotics import GravityForces
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None, dh_config_file=None, pox_config_file=None, camera_extrinsic_file=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        # self.camera.intrinsic_matrix = np.reshape(np.array([917.5701927,    0.,         662.45090881,   0.,         913.11787224,    352.30931891,   0.,           0.,           1.        ]),(3,3))
        self.camera.intrinsic_matrix = np.reshape(np.array([918.3599853515625, 0.0, 661.1923217773438, 0.0, 919.1538696289062, 356.59722900390625, 0.0, 0.0, 1.0]),(3,3))

        if (camera_extrinsic_file is not None):
            self.camera.loadCameraExtrinsic(camera_extrinsic_file)
        else:
            self.camera.extrinsic_matrix =np.array([[ 1,   0,   0,    45 ],
                                                    [ 0,  -1,   0,   165 ],
                                                    [ 0,   0,  -1,   967 ],
                                                    [ 0,   0,   0,     1 ]])
        self.camera.distortion_coeffs = np.array([0.15564486384391785, -0.48568257689476013, -0.0019681642297655344, 0.0007267732871696353, 0.44230175018310547])
        # self.camera.distortion_coeffs = np.array([ 0.07636514, -0.1292355,  -0.00093855,  0.00284562,  0.        ])
        print("Creating rx arm...")
        if (pox_config_file is not None):
            self.rxarm = RXArm(pox_config_file=pox_config_file)
        elif (dh_config_file is not None):
            self.rxarm = RXArm(dh_config_file=dh_config_file)
        else:
            self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.MousePressMoveBlock

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())

        #User Buttons
        self.ui.btnUser1.setText("Calibrate")
        self.ui.btnUser1.clicked.connect(partial(nxt_if_arm_init, 'calibrate'))
        self.ui.btnUser2.setText('Open Gripper')
        self.ui.btnUser2.clicked.connect(lambda: self.rxarm.open())
        self.ui.btnUser3.setText('Close Gripper')
        self.ui.btnUser3.clicked.connect(lambda: self.rxarm.close())
        self.ui.btnUser4.setText('Execute')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'execute'))
        self.ui.btnUser5.setText('Record Waypoint')
        self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'record'))
        self.ui.btnUser6.setText('Playback Waypoints')
        self.ui.btnUser6.clicked.connect(partial(nxt_if_arm_init, 'playback'))
        self.ui.btnUser7.setText('Clear Waypoints')
        self.ui.btnUser7.clicked.connect(partial(nxt_if_arm_init, 'clear'))
        self.ui.btnUser12.setText('Save Images')
        self.ui.btnUser12.clicked.connect(self.camera.saveImage)

        self.ui.btn_task1.clicked.connect(lambda: self.sm.set_next_state('event1'))
        self.ui.btn_task2.clicked.connect(lambda: self.sm.set_next_state('event2'))

        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()

    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    ### TODO: output the rest of the orientation according to the convention chosen
    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str("%+.2f mm" % (pos[0])))
        self.ui.rdoutY.setText(str("%+.2f mm" % (pos[1])))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (pos[2])))
        self.ui.rdoutPhi.setText(str("%+.2f rad" % (pos[3])))
        self.ui.rdoutTheta.setText(str("%+.2f rad" % (pos[4])))
        self.ui.rdoutPsi.setText(str("%+.2f rad" % (pos[5])))

    @pyqtSlot(QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, block_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(block_image))

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)

    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

                    TODO: after implementing workspace calibration display the world coordinates the mouse points to in the RGB
                    video image.

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        pt = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            K = self.camera.intrinsic_matrix
            H_camera_to_world = self.camera.extrinsic_matrix

            z = self.camera.DepthFrameRaw[pt.y()][pt.x()]
            self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                             (pt.x(), pt.y(), z))
            
            object_position_camera = z * np.matmul(np.linalg.inv(K),np.array([pt.x(), pt.y(), 1]).T)
            object_position_world = np.matmul(H_camera_to_world, np.concatenate([object_position_camera,[1]]))
            self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                             (object_position_world[0], object_position_world[1], object_position_world[2]))

    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        pt = mouse_event.pos()
        self.camera.last_click[0] = pt.x()
        self.camera.last_click[1] = pt.y()
        self.camera.new_click = True
        # print(self.camera.last_click)

    def MousePressMoveBlock(self, mouse_event):
        """!
        @brief Grab at the first click and drop at the second click

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        pt = mouse_event.pos()
        self.camera.last_click[0] = pt.x()
        self.camera.last_click[1] = pt.y()
        
        if self.camera.DepthFrameRaw.any() != 0:
            K = self.camera.intrinsic_matrix
            H_camera_to_world = self.camera.extrinsic_matrix

            z = self.camera.DepthFrameRaw[pt.y()][pt.x()]

            pick_from_side = False
            
            object_position_camera = z * np.matmul(np.linalg.inv(K),np.array([pt.x(), pt.y(), 1]).T)
            object_position_world = np.matmul(H_camera_to_world, np.concatenate([object_position_camera,[1]]))
            object_position_arm = transformation_from_world_to_arm(object_position_world)
            arm_x = object_position_arm[0]
            arm_y = object_position_arm[1]
            # end effector pose in arm base frame
            if(np.hypot(object_position_arm[0], object_position_arm[1]) > np.hypot(250,275)):
                theta = np.arctan2(object_position_arm[1], object_position_arm[0])
                T = np.array([
                    [np.cos(theta), -np.sin(theta), 0, object_position_arm[0]],
                    [np.sin(theta), np.cos(theta), 0, object_position_arm[1]],
                    [0, 0, 1, object_position_arm[2]],
                    [0, 0, 0, 1]])

                Outter_area = True
            else:
                T = np.array([
                    [0, 0, 1, object_position_arm[0]],
                    [0, 1, 0, object_position_arm[1]],
                    [-1, 0, 0, object_position_arm[2]],
                    [0, 0, 0, 1]]) # destination

                Outter_area = False

            joint_angle_guess = self.rxarm.get_positions()
            cur_pose = FK_Baseframe(joint_angle_guess, self.rxarm.M_matrix, self.rxarm.S_list)
            total_dis = T[:,3] - cur_pose[:,3]
            total_dis_norm = np.linalg.norm(total_dis)
            print('total dis',total_dis, total_dis_norm)
            if not Outter_area:
                T[1,3] *= 1.04
                T[0,3] = T[0,3]*1.02 - 0.5
            num_mid_points = int(total_dis_norm/150)
            temp_T = cur_pose.copy()
            temp_T[2,3] += 100
            desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.rxarm.S_list, self.rxarm.M_matrix, temp_T, joint_angle_guess, 0.01, 0.001,self.rxarm.resp.upper_joint_limits, self.rxarm.resp.lower_joint_limits)
            if IK_flag:
                self.rxarm.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents, sleep_move_time=True)
                print('lift up the end effector')
            else:
                rospy.logerr("Something wrong with the IK")

            # just calculate the mid points
            # desired_joint_angle = joint_angle_guess
            # for i in range(num_mid_points):
            #     cur_pose = FK_Baseframe(desired_joint_angle, self.rxarm.M_matrix, self.rxarm.S_list)
            #     if(i == num_mid_points-1):
            #         temp_T = T.copy()
            #         temp_T[2,3] += 100
            #         desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.rxarm.S_list, self.rxarm.M_matrix, temp_T, desired_joint_angle, 0.01, 0.001,self.rxarm.resp.upper_joint_limits, self.rxarm.resp.lower_joint_limits)
            #         if IK_flag:
            #             self.rxarm.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents, sleep_move_time=True)
            #             print('final mid points arrived', i)
            #         else:
            #             rospy.logerr("Something wrong with the IK")
            #     else:
            #         temp_T[0:2,3] = cur_pose[0:2,3] + total_dis[0:2]/(num_mid_points+1)
            #         if temp_T[0,3]<75 and abs(temp_T[1,3])<50:
            #             temp_T[0,3] += 150
            #         desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.rxarm.S_list, self.rxarm.M_matrix, temp_T, desired_joint_angle, 0.01, 0.001,self.rxarm.resp.upper_joint_limits, self.rxarm.resp.lower_joint_limits)
            #         if IK_flag:
            #             print('mid points arrived', i)
            #         else:
            #             rospy.logerr("Something wrong with the IK")



            # actually go through the mid points
            # for i in range(num_mid_points):
            #     joint_angle_guess = self.rxarm.get_positions()
            #     cur_pose = FK_Baseframe(joint_angle_guess, self.rxarm.M_matrix, self.rxarm.S_list)
            #     if(i == num_mid_points-1):
            #         temp_T = T.copy()
            #         temp_T[2,3] += 100
            #     else:
            #         temp_T[0:2,3] = cur_pose[0:2,3] + total_dis[0:2]/(num_mid_points+1)
            #         if temp_T[0,3]<75 and abs(temp_T[1,3])<50:
            #             temp_T[0,3] += 150
            #     desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.rxarm.S_list, self.rxarm.M_matrix, temp_T, joint_angle_guess, 0.01, 0.001,self.rxarm.resp.upper_joint_limits, self.rxarm.resp.lower_joint_limits)
            #     if IK_flag:
            #         self.rxarm.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents, sleep_move_time=False)
            #         print('mid points arrived', i)
            #     else:
            #         rospy.logerr("Something wrong with the IK")

            # go through the home position
            
            # if arm_x > 125 and arm_y>0:
            #     joint_angle_guess = np.array([np.pi/4,0,0,0,0])
            # elif arm_x <125 and arm_y>0:
            #     joint_angle_guess = np.array([np.pi/2,0,0,0,0])
            # elif arm_x>125 and arm_y<0:
            #     joint_angle_guess = np.array([-np.pi/4,0,0,0,0])
            # elif arm_x<125 and arm_y<0:
            #     joint_angle_guess = np.array([-np.pi/2,0,0,0,0])
            # else:
            #     joint_angle_guess = np.zeros(5)
            base_angle = np.arctan2(arm_y, arm_x)
            joint_angle_guess = np.array([base_angle,0,0,0,0])
                
            print('joint guess', joint_angle_guess)
            temp_T = T.copy()
            temp_T[:,3] = T[:,3] #.copy()
            temp_T[2,3] += 100
            print('final mid point', temp_T[:,3])
            desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.rxarm.S_list, self.rxarm.M_matrix, temp_T, joint_angle_guess, 0.01, 0.001,self.rxarm.resp.upper_joint_limits, self.rxarm.resp.lower_joint_limits)
            if IK_flag:
                self.rxarm.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents, sleep_move_time=True)
                print('final mid points arrived')
            else:
                rospy.logerr("Something wrong with the IK")

            
            # temp_T = T.copy() 
            # temp_T[:,3] = cur_pose[:,3]
            # temp_T[2,3] += 100
            # desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.rxarm.S_list, self.rxarm.M_matrix, temp_T, joint_angle_guess, 0.01, 0.001,self.rxarm.resp.upper_joint_limits, self.rxarm.resp.lower_joint_limits)
            # print('IK flag and joint angles for mid points', IK_flag, desired_joint_angle)
            # if IK_flag:
            #     self.rxarm.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents, sleep_move_time=False)
            #     print('mid points arrived')
            # else:
            #     rospy.logerr("Something wrong with the IK")
            
        if self.camera.new_click == False:
            joint_angle_guess = self.rxarm.get_positions()
            T_grab = T.copy()
            T_grab[2,3] += 15
            
            print(T_grab[:,3])
            desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.rxarm.S_list, self.rxarm.M_matrix, T_grab, joint_angle_guess, 0.01, 0.001,self.rxarm.resp.upper_joint_limits, self.rxarm.resp.lower_joint_limits)
            # extra_torque = GravityForces(desired_joint_angle, np.array([0, 0, -9800]), self.rxarm.Mlist, self.rxarm.Glist, self.rxarm.S_list.T)
            # with open('extra_torque.txt', 'a') as outfile1:    
            #     np.savetxt(outfile1, [extra_torque], fmt='%f', delimiter= ',')
            # print('extra_torque', extra_torque)
            if IK_flag:
                
                self.rxarm.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents)
                actual_angle = self.rxarm.get_positions()
                angle_difference = desired_joint_angle - actual_angle
                with open('angle_difference.txt', 'a') as outfile2:    
                    np.savetxt(outfile2, [angle_difference], fmt='%f', delimiter= ',')
                print('angle difference', angle_difference)
            else:
                rospy.logerr("Something wrong with the IK")
            self.rxarm.close()
            self.camera.new_click = True
        else:
            joint_angle_guess = self.rxarm.get_positions()
            T_drop = T.copy()
            T_drop[2,3] += 60
            # T_drop[1,3] *= 1.04
            # T_drop[0,3] = T_drop[0,3]*1.02 - 0.5
            print(T_drop[:,3])
            desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.rxarm.S_list, self.rxarm.M_matrix, T_drop, joint_angle_guess, 0.01, 0.001,self.rxarm.resp.upper_joint_limits, self.rxarm.resp.lower_joint_limits)
            # extra_torque = GravityForces(desired_joint_angle, np.array([0, 0, -9800]), self.rxarm.Mlist, self.rxarm.Glist, self.rxarm.S_list.T)
            # with open('extra_torque.txt', 'a') as outfile1:    
            #     np.savetxt(outfile1, [extra_torque], fmt='%f', delimiter= ',')
            # print('extra_torque', extra_torque)
            if IK_flag:
                self.rxarm.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents)
                actual_angle = self.rxarm.get_positions()
                angle_difference = desired_joint_angle - actual_angle
                with open('angle_difference.txt', 'a') as outfile2:    
                    np.savetxt(outfile2, [angle_difference], fmt='%f', delimiter= ',')
                print('angle difference', angle_difference)
            else:
                rospy.logerr("Something wrong with the IK")
            self.rxarm.open()
            self.camera.new_click = False

                





    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')


### TODO: Add ability to parse POX config file as well
def main(args=None):
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui(pox_config_file=args['poxconfig'],camera_extrinsic_file=args['extrinsic'])
    app_window.show()
    sys.exit(app.exec_())


# Run main if this file is being run directly
### TODO: Add ability to parse POX config file as well
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d",
                    "--dhconfig",
                    required=False,
                    help="path to DH parameters csv file")
    ap.add_argument("-p",
                    "--poxconfig",
                    required=False,
                    help="path to PoX parameters csv file")
    ap.add_argument("-e",
                    "--extrinsic",
                    required=False,
                    help="path to camera extrinsic txt file")

    main(args=vars(ap.parse_args()))
