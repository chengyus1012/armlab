"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from functools import partial
from kinematics import FK_dh, FK_pox, get_pose_from_T
import time
import csv
from builtins import super
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
from interbotix_robot_arm import InterbotixRobot
from interbotix_descriptions import interbotix_mr_descriptions as mrd
from config_parse import *
from sensor_msgs.msg import JointState
import rospy
import scipy
"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixRobot):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None, pox_config_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_name="rx200", mrd=mrd)
        self.joint_names = self.resp.joint_names
        print(self.joint_names)
        self.num_joints = 5
        # Gripper
        self.gripper_state = True
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        self.dh_params = []
        self.dh_config_file = dh_config_file
        if (dh_config_file is not None):
            self.dh_params = self.parse_dh_param_file(dh_config_file)
        #POX params
        self.M_matrix = []
        self.S_list = []
        self.pox_config_file = pox_config_file
        if (pox_config_file is not None):
            self.M_matrix, self.S_list = load_pox_param_file(pox_config_file)

        print(self.M_matrix, self.S_list)
        self.max_angular_vel = 0.8 # rad/s

    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        rospy.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.set_gripper_pressure(1.0)
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.open()
        self.initialized = True
        return self.initialized

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def set_positions(self, joint_positions):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        self.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)

    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_off(self.joint_names)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_on(self.joint_names)

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb


#   @_ensure_initialized
    def construct_se3_matrix(self, vector):
        w = np.zeros((3,3))
        w[0,1] = -vector[2]
        w[1,0] = vector[2]
        w[0,2] = vector[1]
        w[2,0] = -vector[1]
        w[1,2] = -vector[2]
        w[2,1] = vector[2]
        v = vector[3:].reshape((3,1))
        S = np.block([[w,v],[np.zeros((1,4))]])
        return S



    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as [x, y, z, phi] or as needed.
        """
        cur_pos = np.array(self.rxarm.get_positions())

        temp = np.array((4,4))
        for i in range(5):
            temp = np.matmul(temp,scipy.linalg.expm(self.construct_se3_matrix(self.S_list[i,:])*cur_pos[i]))
        ans = np.matmul(temp, self.M_matrix)
         
        
        return [0, 0, 0, 0]

    @_ensure_initialized
    def get_wrist_pose(self):
        """!
        @brief      TODO Get the wrist pose.

        @return     The wrist pose as [x, y, z, phi] or as needed.
        """
        return [0, 0, 0, 0]

    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        print("Parsing PoX config file...")
        pox_params = load_pox_param_file(self.pox_config_file)
        print("PoX config file parse exit.")

        return pox_params

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        dh_params = parse_dh_param_file(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params

    def close(self):
        self.torque_joints_on([self.joint_names[-1]])
        self.close_gripper()
        self.gripper_state = True

    def open(self):
        self.torque_joints_on([self.joint_names[-1]])
        self.open_gripper()
        self.gripper_state = False


class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        rospy.Subscriber('/rx200/joint_states', JointState, self.callback)
        rospy.sleep(0.5)

    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        # for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.velocity_fb))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)

    def run(self):
        """!
        @brief      Updates the RXArm Joints at a set rate if the RXArm is initialized.
        """
        while True:

            rospy.spin()

if __name__ == '__main__':
    rxarm = RXArm()
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
        rxarm.initialize()

        rxarm.go_to_home_pose()
        rxarm.set_gripper_pressure(0.5)
        rxarm.set_joint_positions(joint_positions,
                                  moving_time=2.0,
                                  accel_time=0.5,
                                  blocking=True)
        rxarm.close()
        rxarm.go_to_home_pose()
        rxarm.open()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")
