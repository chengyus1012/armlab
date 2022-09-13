"""!
The state machine that implements the logic.
"""
from cmath import sqrt
from turtle import pos
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
import cv2
from scipy.spatial.transform import Rotation
from apriltag_ros.msg import AprilTagDetectionArray

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
        self.tag_positions = np.array([[-250, -25, 0], # 1
                               [ 250, -25, 0], # 2
                               [ 250, 275, 0], # 3
                               [-250, 275, 0]])# 4 # in world frame
        self.K = np.reshape(np.array([918.3599853515625, 0.0, 661.1923217773438, 0.0, 919.1538696289062, 356.59722900390625, 0.0, 0.0, 1.0]),(3,3))
        self.apriltag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.apriltag_callback)

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

                    TODO: Add states and funcitons as needed.
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

        # IDs: BL = 1, BR = 2, TR = 3, TL = 4 (BR unstable)
        points_camera = np.zeros(self.tag_positions.shape)
        for tag in self.latest_tags.detections:
            id = tag.id[0]
            tag_position = np.array([tag.pose.pose.pose.position.x, tag.pose.pose.pose.position.y, tag.pose.pose.pose.position.z]) * 1000
            points_camera[id - 1, :] = tag_position

        A_svd = self.recover_homogeneous_transform_svd(self.tag_positions, points_camera)

        self.camera.extrinsic_matrix = A_svd

        # A_affine_cv = self.recover_homogeneous_affine_opencv(
        #     points_camera.astype(np.float32), self.tag_positions.astype(np.float32))
        # self.camera.extrinsic_matrix = A_affine_cv

        # A_affine = self.recover_homogenous_affine_transformation(self.tag_positions[0:3], points_camera[0:3])
        # self.camera.extrinsic_matrix = A_affine

        rospy.logerr("Matrix: \n%s",self.camera.extrinsic_matrix)
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
        rospy.sleep(1)

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