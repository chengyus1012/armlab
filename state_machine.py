"""!
The state machine that implements the logic.
"""
from cmath import sqrt
from turtle import pos
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
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
        self.tag_positions = np.array([[-250, -25, 0],
                               [ 250, -25, 0],
                               [ 250, 275, 0],
                               [-250, 275, 0]]) # in world frame

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
        
        # IDs: BL = 1, BR = 2, TR = 3, TL = 4 (BR unstable)
        plane_positions = [0] * 4
        for tag in self.latest_tags.detections:
            id = tag.id[0]
            position = np.array([tag.pose.pose.pose.position.x, tag.pose.pose.pose.position.y, tag.pose.pose.pose.position.z]) * 1000
            if id == 1:
                tag_position_in_world = self.tag_positions[id - 1]
                tag_position_relative_to_camera = position
            plane_positions[id - 1] = position

        vec_4_to_1 = plane_positions[0] - plane_positions[3]
        vec_4_to_1 = vec_4_to_1 / np.linalg.norm(vec_4_to_1)
        vec_4_to_3 = plane_positions[2] - plane_positions[3]
        vec_4_to_3 = vec_4_to_3 / np.linalg.norm(vec_4_to_3)
        normal_vec = np.cross(vec_4_to_1,vec_4_to_3)
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        
        # Rotation.from_rotvec(normal_vec)
        # c1 = sqrt(normal_vec[0] ** )
        rot_matrix = np.array([vec_4_to_1,
                               normal_vec,
                               vec_4_to_3])

        H_rot_camera_to_tag_plane = np.block([[rot_matrix,np.zeros((3,1))],[np.zeros((1,3)),1]])
        H_world_to_tag_plane = np.array([   [ 0,  -1,   0,   tag_position_in_world[0] ],
                                            [ 1,   0,   0,   tag_position_in_world[1] ],
                                            [ 0,   0,   1,   tag_position_in_world[2] ],
                                            [ 0,   0,   0,     1 ]])
        # H_camera_to_tag = np.array([[ 1,   0,   0,   tag_position_relative_to_camera[0] ],
        #                             [ 0,  -1,   0,   tag_position_relative_to_camera[1] ],
        #                             [ 0,   0,  -1,   tag_position_relative_to_camera[2] ],
        #                             [ 0,   0,   0,     1 ]])
        rospy.logerr("Shapes %s %s", rot_matrix.shape,tag_position_relative_to_camera.shape)
        H_camera_to_tag = np.block([[rot_matrix,np.reshape(tag_position_relative_to_camera,(3,1))],[np.zeros((1,3)),1]])
        H_world_to_camera = np.matmul(H_world_to_tag_plane, np.linalg.inv(H_camera_to_tag))
        rospy.logerr("Matrix: \n%s",H_world_to_camera)
        self.status_message = "Calibration - Completed Calibration"

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