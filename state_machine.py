"""!
The state machine that implements the logic.
"""
from __future__ import print_function
from asyncore import write
from cmath import sqrt
from turtle import pos
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
import cv2
import math
import pandas as pd 
from scipy.spatial.transform import Rotation
from apriltag_ros.msg import AprilTagDetectionArray
from rxarm import D2R
from utility_functions import *

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [[-np.pi/2,       -0.5,      -0.3,      0.0,       0.0   ],
                            [0.75*-np.pi/2,   0.5,       0.3,      0.0,      np.pi/2],
                            [0.5*-np.pi/2,   -0.5,      -0.3,     np.pi/2,    0.0   ],
                            [0.25*-np.pi/2,   0.5,       0.3,      0.0,      np.pi/2],
                            [0.0,             0.0,       0.0,      0.0,       0.0   ],
                            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,      np.pi/2],
                            [0.5*np.pi/2,     0.5,       0.3,     np.pi/2,    0.0   ],
                            [0.75*np.pi/2,   -0.5,      -0.3,      0.0,      np.pi/2],
                            [np.pi/2,         0.5,       0.3,      0.0,       0.0   ],
                            [0.0,             0.0,       0.0,      0.0,       0.0   ]]
        self.recorded_positions = []
        self.tag_positions = np.array([[-250, -25, 0], # 1
                               [ 250, -25, 0], # 2
                               [ 250, 275, 0], # 3
                               [-250, 275, 0], # 4
                               [425,  400, 151], # 5
                               [-425, 400, 241], # 6 
                               [-425, -100, 92]])# 7 # in world frame
        self.K = np.reshape(np.array([917.5701927, 0., 662.45090881, 0., 913.11787224, 352.30931891, 0., 0., 1.]),(3,3))
        self.D = [ 0.07636514, -0.1292355,  -0.00093855,  0.00284562,  0.        ]
        self.apriltag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.apriltag_callback)
        self.sample_rate = 20.0 # in Hz

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "record":
            self.record()
        
        if self.next_state == "playback":
            self.playback()
        
        if self.next_state == "clear":
            self.clear()

        if self.next_state == "event1":
            self.event1()

        if self.next_state == "event2":
            self.event2()

        if self.next_state == "event3":
            self.event3()

        if self.next_state == "event4":
            self.event4()

        if self.next_state == "event5":
            self.event5()

    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"
        for pt in self.waypoints:
            if self.next_state == "estop":
                return
            self.rxarm.set_positions(pt)
            rospy.sleep(self.rxarm.moving_time)
        self.next_state = "idle"

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        if len(self.latest_tags.detections) < 4:
            self.status_message = "Not enough tags visible (%d of 4)" % len(self.latest_tags)
            return


        ### None of these worked well (all fairly innacurate), will have to change to solvePNP
        print("MAT\n", self.camera.intrinsic_matrix)
        # IDs: BL = 1, BR = 2, TR = 3, TL = 4 (BR unstable)
        points_camera = np.zeros(self.tag_positions.shape)
        pixel_coords = np.zeros(self.tag_positions.shape, dtype=int)
        for tag in self.latest_tags.detections:
            id = tag.id[0]
            tag_position = np.array([tag.pose.pose.pose.position.x, tag.pose.pose.pose.position.y, tag.pose.pose.pose.position.z]) * 1000
            pixel_coords[id - 1, :] = np.matmul(np.block([(1 / (tag_position[2])) * self.camera.intrinsic_matrix, np.zeros((3,1))]), np.append(tag_position,1))
            z =self.camera.DepthFrameRaw[pixel_coords[id - 1, 1],pixel_coords[id - 1, 0]]
            points_camera[id - 1, :] =  np.matmul(z * np.linalg.inv(self.camera.intrinsic_matrix), pixel_coords[id - 1, :])
        
        print("Pixels:\n",pixel_coords)
        A_pnp = self.recover_homogenous_transform_pnp(pixel_coords[:,:-1].astype(np.float32),self.tag_positions.astype(np.float32), self.camera.intrinsic_matrix)
        self.camera.extrinsic_matrix = np.linalg.inv(A_pnp)
        np.savetxt('config/latest_calibration.txt',self.camera.extrinsic_matrix)
        # self.camera.extrinsic_matrix[2,3] += 10
        # A_svd = self.recover_homogeneous_transform_svd(self.tag_positions, points_camera)

        # self.camera.extrinsic_matrix = np.linalg.inv(A_svd)

        # A_affine_cv = self.recover_homogeneous_affine_opencv(
        #     points_camera.astype(np.float32), self.tag_positions.astype(np.float32))
        # self.camera.extrinsic_matrix = A_affine_cv

        # A_affine = self.recover_homogenous_affine_transformation(self.tag_positions[0:3], points_camera[0:3])
        # self.camera.extrinsic_matrix = A_affine

        self.status_message = "Calibration - Completed Calibration"

        # # Old Method
        # plane_positions = [0] * 4
        # for tag in self.latest_tags.detections:
        #     id = tag.id[0]
        #     position = np.array([tag.pose.pose.pose.position.x, tag.pose.pose.pose.position.y, tag.pose.pose.pose.position.z]) * 1000
        #     if id == 1:
        #         tag_position_in_world = self.tag_positions[id - 1]
        #         tag_position_relative_to_camera = position
        #     plane_positions[id - 1] = position

        # vec_4_to_1 = plane_positions[0] - plane_positions[3]
        # vec_4_to_1 = vec_4_to_1 / np.linalg.norm(vec_4_to_1)
        # vec_4_to_3 = plane_positions[2] - plane_positions[3]
        # vec_4_to_3 = vec_4_to_3 / np.linalg.norm(vec_4_to_3)
        # normal_vec = np.cross(vec_4_to_1,vec_4_to_3)
        # normal_vec = normal_vec / np.linalg.norm(normal_vec)
        
        # # # Rotation.from_rotvec(normal_vec)
        # # # c1 = sqrt(normal_vec[0] ** )
        # # rot_matrix = np.array([vec_4_to_1,
        # #                        normal_vec,
        # #                        vec_4_to_3])

        # # H_rot_camera_to_tag_plane = np.block([[rot_matrix,np.zeros((3,1))],[np.zeros((1,3)),1]])
        # # H_world_to_tag_plane = np.array([   [ 0,  -1,   0,   tag_position_in_world[0] ],
        # #                                     [ 1,   0,   0,   tag_position_in_world[1] ],
        # #                                     [ 0,   0,   1,   tag_position_in_world[2] ],
        # #                                     [ 0,   0,   0,     1 ]])
        # # # H_camera_to_tag = np.array([[ 1,   0,   0,   tag_position_relative_to_camera[0] ],
        # # #                             [ 0,  -1,   0,   tag_position_relative_to_camera[1] ],
        # # #                             [ 0,   0,  -1,   tag_position_relative_to_camera[2] ],
        # # #                             [ 0,   0,   0,     1 ]])

        # # rospy.logerr("Shapes %s %s", rot_matrix.shape,tag_position_relative_to_camera.shape)
        # # H_camera_to_tag = np.block([[rot_matrix,np.reshape(tag_position_relative_to_camera,(3,1))],[np.zeros((1,3)),1]])
        # # H_world_to_camera = np.matmul(H_world_to_tag_plane, np.linalg.inv(H_camera_to_tag))
        # # rospy.logerr("Matrix: \n%s",H_world_to_camera)
        # # self.status_message = "Calibration - Completed Calibration"

        # # ############################ chengyu version ########################################
        # # # unit vectors of x,y,z axis of the tag frame in camera frame, representing rotation from camera to tag
        # # # and the displacement should be a vector pointing from camera frame to tag frame
        # rot_matrix_tag_to_camera = np.array([vec_4_to_3,
        #                        vec_4_to_1, 
        #                        normal_vec,
        #                        ])

        # H_tag_plane_to_world = np.array([   [ 1,   0,   0,   tag_position_in_world[0] ],
        #                                     [ 0,  -1,   0,   tag_position_in_world[1] ],
        #                                     [ 0,   0,   1,   tag_position_in_world[2] ],
        #                                     [ 0,   0,   0,     1 ]])

        # H_tag_to_camera = np.block([[rot_matrix_tag_to_camera, np.reshape(tag_position_relative_to_camera,(3,1))],[np.zeros((1,3)),1]])
        # # The first result we calculated is H_camera_to_world = H_tag_to_world * H_camera_tag,
        # # and that's why they are so close!
        # # We can still calculate H_camera_to_world
        # H_camera_to_world = np.matmul(H_tag_plane_to_world, np.linalg.inv(H_tag_to_camera))
        
        # self.camera.extrinsic_matrix = H_camera_to_world
        # ############################ end of the code ########################################


    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        pass

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"
    
    def record(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "record"
        self.status_message = "Recording waypoint"
        if not self.rxarm.initialized:
            print('Arm is not initialized')
            self.status_message = "State: Arm is not initialized!"
            self.next_state = "idle"
            return
        
        current_position = [self.rxarm.get_positions(), self.rxarm.gripper_state]
        self.recorded_positions.append(current_position)
        self.next_state = "idle"

    def playback(self):
        """!
        @brief      Go through all waypoints
        """
        self.status_message = "State: Playback - Executing recorded waypoints"
        self.current_state = "playback"
        rate = rospy.Rate(self.sample_rate)
        sleep_time = 1.0 / self.sample_rate
        rate.sleep()
        joint_angle_data = []
        end_position_data = []
        zero_time = rospy.Time().to_nsec()
        # pd.DataFrame({'Time': [], 'J1' : [], 'J2' : [], 'J3' : [], 'J4' : [], 'J5' : [], 'End' : })
        for pt in self.recorded_positions:
            if self.next_state == "estop":
                return
            cur_pos = np.array(self.rxarm.get_positions())
            max_delta = max(abs(np.array(pt[0]) - cur_pos))
            move_time = max_delta / self.rxarm.max_angular_vel
            num_data_points = int(math.ceil(move_time / sleep_time))
            self.rxarm.set_joint_positions(pt[0],moving_time = move_time, accel_time = move_time/3, blocking=False)
            for i in range(num_data_points):
                cur_time = rospy.Time.now().to_nsec()
                cur_pos = np.array(self.rxarm.get_positions())
                cur_joint_data = np.concatenate([[cur_time - zero_time],cur_pos])
                end_pos = self.rxarm.get_ee_pose()
                joint_angle_data.append(cur_joint_data)
                end_position_data.append(end_pos)
                rate.sleep()
            # rospy.sleep(move_time)
            if pt[1] == True: # Closed
                self.rxarm.close()
            else:
                self.rxarm.open()
        np.savetxt('joint_data.txt', joint_angle_data, fmt='%f', delimiter= ',')
        write_to_file(path="end_data.txt", array=end_position_data)
        self.next_state = "idle"
    
    def clear(self):
        self.status_message = "State: Clearing waypoints"
        self.current_state = "clear"
        self.recorded_positions = []
        self.next_state = "idle"

    def event1(self):
        self.status_message = "State: Performing event 1"
        self.current_state = "event1"
        
        small_store_positions = np.array([[-250, -125, 0], # 1
                                            [-200, -125, 0], # 2
                                            [-150, -125, 0], # 2
                                            [-100, -125, 0], # 2
                                            [-275, -50, 0], # 2
                                            [-225, -50, 0], # 2
                                            [-175, -50, 0], # 2
                                            [-125, -50, 0]])# 7 # in world frame

        large_store_positions = small_store_positions.copy()
        large_store_positions[:,0] *= -1
        task_complete = False
        curr_large_store_current_idx = 0
        curr_small_store_current_idx = 0

        self.rxarm.startup()

        while not task_complete:
            print("Loop")
            self.rxarm.stow_arm()

            self.current_ee_pose = self.rxarm.get_ee_pose()[:3,3]
            aggregate_blocks = []
            for i in range(5):
                current_blocks = self.camera.detect_blocks(self.current_ee_pose)
                self.current_blocks = current_blocks
                # aggregate_blocks.append(current_blocks)

            self.current_blocks = filter(lambda block: block.top_face_position[1] > 0,self.current_blocks)
            print(len(self.current_blocks), "blocks detected")
            if(len(self.current_blocks) == 0):
                task_complete = True
                break

            self.current_blocks.sort(key=lambda block: math.sqrt(block.top_face_position[0]**2 + block.top_face_position[1]**2) )
            
            vertically_reachable_blocks = filter(lambda block: self.rxarm.reachable(block.top_face_position, vertical=True, above=True, is_large=block.is_large), self.current_blocks)
            print(len(vertically_reachable_blocks), "vertically reachable blocks")
            if(len(vertically_reachable_blocks) == 0):
                selected_blocks = self.current_blocks
                approach_vertically = False
            else:
                selected_blocks = vertically_reachable_blocks
                approach_vertically = True
            for block in selected_blocks:
                print(block)
            for block in selected_blocks:
                print("Going to block", block.color,"at",block.top_face_position,block.angle)
                self.rxarm.go_to_safe()
                success = self.rxarm.move_above(block.top_face_position, block.angle, vertical=approach_vertically)
                if (not success):
                    continue
                self.rxarm.grab(block.top_face_position,block.angle, block.is_large, vertical=approach_vertically)
                print("grab finished")
                
                self.rxarm.go_to_safe()
                print("go to safe")

                if block.is_large:
                    print("Placing at",large_store_positions[curr_large_store_current_idx,:])
                    self.rxarm.move_above(large_store_positions[curr_large_store_current_idx,:], 90*D2R, vertical=True)
                    self.rxarm.place_on(large_store_positions[curr_large_store_current_idx,:], 90*D2R, vertical=True)
                    curr_large_store_current_idx += 1
                    curr_large_store_current_idx %= len(large_store_positions)
                else:
                    print("Placing at",small_store_positions[curr_small_store_current_idx,:])

                    self.rxarm.move_above(small_store_positions[curr_small_store_current_idx,:], 90*D2R, vertical=True)
                    self.rxarm.place_on(small_store_positions[curr_small_store_current_idx,:], 90*D2R, vertical=True)
                    curr_small_store_current_idx += 1
                    curr_small_store_current_idx %= len(small_store_positions)

                break    
                if self.next_state == "estop":
                    return



            if self.next_state == "estop":
                return

        self.status_message = "State: Event 1 complete"
        self.next_state = "idle"

    def event2(self):
        self.status_message = "State: Performing event 2"
        self.current_state = "event2"

        self.status_message = "State: Event 2 complete"
        self.next_state = "idle"

    def event3(self):
        self.status_message = "State: Performing event 3"
        self.current_state = "event3"

        self.status_message = "State: Event 3 complete"
        self.next_state = "idle"

    def event4(self):
        self.status_message = "State: Performing event 4"
        self.current_state = "event4"

        self.status_message = "State: Event 4 complete"
        self.next_state = "idle"

    def event5(self):
        self.status_message = "State: Performing event 5"
        self.current_state = "event5"

        self.status_message = "State: Event 5 complete"
        self.next_state = "idle"
        
    def apriltag_callback(self, tags):
        self.latest_tags = tags

    def recover_homogeneous_transform_svd(self, m, d):
        ''' 
        finds the rigid body transform that maps m to d: 
        d == np.dot(m,R) + T
        http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
        '''
        # calculate the centroid for each set of points
        d_bar = np.sum(d, axis=0) / np.shape(d)[0]
        m_bar = np.sum(m, axis=0) / np.shape(m)[0]

        # we are using row vectors, so tanspose the first one
        # H should be 3x3, if it is not, we've done this wrong
        H = np.dot(np.transpose(d - d_bar), m - m_bar)
        [U, S, V] = np.linalg.svd(H)

        R = np.matmul(V, np.transpose(U))
        # if det(R) is -1, we've made a reflection, not a rotation
        # fix it by negating the 3rd column of V
        if np.linalg.det(R) < 0:
            V = [1, 1, -1] * V
            R = np.matmul(V, np.transpose(U))
        T = d_bar - np.dot(m_bar, R)
        return np.transpose(np.column_stack((np.row_stack((R, T)), (0, 0, 0, 1))))

    def recover_homogeneous_affine_opencv(self, src, dst):
        _, T, _ = cv2.estimateAffine3D(src, dst, confidence=0.99)
        #print(T)
        return np.row_stack((T, (0.0, 0.0, 0.0, 1.0)))

    def recover_homogenous_affine_transformation(self, p, p_prime):
        '''points_transformed_1 = points_transformed_1 = np.dot(
        A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))np.dot(
        A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))
        Find the unique homogeneous affine transformation that
        maps a set of 3 points to another set of 3 points in 3D
        space:

            p_prime == np.dot(p, R) + t

        where `R` is an unknown rotation matrix, `t` is an unknown
        translation vector, and `p` and `p_prime` are the original
        and transformed set of points stored as row vectors:

            p       = np.array((p1,       p2,       p3))
            p_prime = np.array((p1_prime, p2_prime, p3_prime))

        The result of this function is an augmented 4-by-4
        matrix `A` that represents this affine transformation:

            np.column_stack((p_prime, (1, 1, 1))) == \
                np.dot(np.column_stack((p, (1, 1, 1))), A)

        Source: https://math.stackexchange.com/a/222170 (robjohn)
        '''

        # construct intermediate matrix
        Q = p[1:] - p[0]
        Q_prime = p_prime[1:] - p_prime[0]

        # calculate rotation matrix
        R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
                np.row_stack((Q_prime, np.cross(*Q_prime))))

        # calculate translation vector
        t = p_prime[0] - np.dot(p[0], R)

        # calculate affine transformation matrix
        return np.transpose(np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1))))

    def recover_homogenous_transform_pnp(self, image_points, world_points, K):
        '''
        Use SolvePnP to find the rigidbody transform representing the camera pose in
        world coordinates (not working)
        '''
        distCoeffs = self.camera.distortion_coeffs
        image_points = np.ascontiguousarray(image_points).reshape((image_points.shape[0],1,2))
        [_, R_exp, t] = cv2.solvePnP(world_points,
                                    image_points,
                                    K,
                                    distCoeffs,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(R_exp)
        return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))

    def stow_arm(self):
        moving_time = 1.5
        accel_time = 0.75
        self.rxarm.go_to_home_pose(moving_time=moving_time,
                             accel_time=accel_time,
                             blocking=True)
        self.rxarm.go_to_sleep_pose(moving_time=moving_time,
                              accel_time=accel_time,
                              blocking=True)

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)