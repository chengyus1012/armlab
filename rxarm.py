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
from tkinter import SEL
import numpy as np
from functools import partial
from kinematics import FK_dh, FK_pox, get_pose_from_T
import time
import csv
from builtins import super
from PyQt4.QtCore import QThread, pyqtSignal, QTimer, QCoreApplication
from interbotix_robot_arm import InterbotixRobot
from interbotix_descriptions import interbotix_mr_descriptions as mrd
from config_parse import *
from sensor_msgs.msg import JointState
import rospy
from scipy.linalg import expm
from utility_functions import *
from kinematics import *
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
        # self.shoulder_diff = 0.0028790661356
        # self.elbow_diff = 0.001215238360935

        self.shoulder_diff = -3*D2R
        self.elbow_diff = 4*D2R
        self.wran_diff = 5.5*D2R


        
        super().__init__(robot_name="rx200", mrd=mrd)
        self.joint_names = self.resp.joint_names
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
        # transformation between links
        self.M01 = np.eye(4)
        self.M12 = np.eye(4)
        self.M23 = np.eye(4)
        self.M34 = np.eye(4)
        self.M45 = np.eye(4)
        self.M56 = np.eye(4)

        self.M01[0:3,3] = np.array([0, 0, 65]) # waist
        self.M12[0:3,3] = np.array([0, 0, 38.91]) # shoulder
        self.M23[0:3, 0:3] = Rx(np.pi)
        self.M23[0:3,3] = np.array([50, 0, 200]) # elbow
        self.M34[0:3,3] = np.array([200, 0, 0]) # wrist angle
        self.M45[0:3, 0:3] = Rx(-np.pi)
        self.M45[0:3,3] = np.array([65, 0, 0]) # wrist rotate
        self.M56[0:3,3] = np.array([43, 0, 0]) # ee_arm

        self.Mlist = np.array([self.M01, self.M12, self.M23, self.M34, self.M45, self.M56])

        # position and orientation of inertial frame in the corresponding link frame
        self.T1 = np.eye(4) # shoulder_link
        self.T1[0:3, 0:3] = Rz(np.pi/2)
        self.T1[0:3,3] = np.array([-0.0000853644, 0.0000173690, 0.0132005000])*1000
        self.m1 = 0.257774 # mass
        self.inertial_vector1 = np.array([0.0002663000, 0.0004428000, 0.0004711000, 0.0000000009, 0.0000000511, 0.0000004416])*1e6
        self.G1 = np.matmul(np.matmul(Adjoint(self.T1), construct_SI_mat(self.inertial_vector1, self.m1)), Adjoint(self.T1).T)

        self.T2 = np.eye(4) # upper_arm_link
        self.T2[0:3, 0:3] = Rz(np.pi/2)
        self.T2[0:3,3] = np.array([0.0119513000, -0.0001169230, 0.1394300000])*1000
        self.m2 = 0.297782 # mass
        self.inertial_vector2 = np.array([0.0017100000, 0.0016310000, 0.0001478000, -0.0000009773, 0.0000020936, 0.0002132000])*1e6
        self.G2 = np.matmul(np.matmul(Adjoint(self.T2), construct_SI_mat(self.inertial_vector2, self.m2)), Adjoint(self.T2).T)

        self.T3 = np.eye(4) # forearm_link
        self.T3[0:3, 0:3] = Rz(np.pi/2)
        self.T3[0:3,3] = np.array([0.1147450000, -0.0000938376, 0.0000000000])*1000
        self.m3 = 0.258863 # mass
        self.inertial_vector3 = np.array([0.0010550000,0.0000642100,0.0010760000,-0.0000018286,0.0000000000,0.0000000000])*1e6
        self.G3 = np.matmul(np.matmul(Adjoint(self.T3), construct_SI_mat(self.inertial_vector3, self.m3)), Adjoint(self.T3).T)

        self.T4 = np.eye(4) # wrist_link
        self.T4[0:3, 0:3] = np.matmul(Rz(np.pi/2), Ry(np.pi))
        self.T4[0:3,3] = np.array([0.0423600000, 0.0000104110, -0.0105770000])*1000
        self.m4 = 0.084957 # mass
        self.inertial_vector4 = np.array([0.0000308200,0.0000282200,0.0000315200,0.0000000191,0.0000000023,0.0000025481])*1e6
        self.G4 = np.matmul(np.matmul(Adjoint(self.T4), construct_SI_mat(self.inertial_vector4, self.m4)), Adjoint(self.T4).T)

        self.T5 = np.eye(4) # gripper_link
        self.T5[0:3, 0:3] = Rz(np.pi/2)
        self.T5[0:3,3] = np.array([0.0216300000, 0.0000000000, 0.0114100000])*1000
        self.m5 = 0.072885 # mass
        self.inertial_vector5 = np.array([0.0000253700, 0.0000183600, 0.0000167400, 0.0000000000, 0.0000000000, 0.0000004340])*1e6
        self.G5 = np.matmul(np.matmul(Adjoint(self.T5), construct_SI_mat(self.inertial_vector5, self.m5)), Adjoint(self.T5).T)

        self.Glist = np.array([self.G1, self.G2, self.G3, self.G4, self.G5])

        self.safe_position = np.array([0.0, -20.0, 70.0, -90.0, 0.0])*D2R


        #POX params
        self.M_matrix = []
        self.S_list = []
        self.pox_config_file = pox_config_file
        if (pox_config_file is not None):
            self.M_matrix, self.S_list = load_pox_param_file(pox_config_file)

        self.max_angular_vel = 0.6 # rad/s

        self.pid_gains = {"waist":          [640, 0, 3600], # waist gains
                          "shoulder":       [1000, 0, 1000], # shoulder gains
                          "elbow":          [1000, 0, 1500], # elbow gains 
                          "wrist_angle":    [800, 0, 1000], # wrist_angle gains
                          "wrist_rotate":   [640, 0, 3600], # wrist_rotate gains 
                          "gripper":        [640, 0, 3600] # gripper gains
        }

        for joint_name in self.joint_names:
            self.set_joint_position_pid_params(joint_name, self.pid_gains[joint_name])

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

    def startup(self):
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

    def set_positions_custom(self, joint_positions, gui_func=None, sleep_move_time = True):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        cur_pos = np.array(self.get_positions())
        max_delta = max(abs(np.array(joint_positions) - cur_pos))
        move_time = max_delta / self.max_angular_vel
        # joint_positions[1]+=self.shoulder_diff
        # joint_positions[2]+=self.elbow_diff
        joint_positions[3]+=self.wran_diff
        self.set_joint_positions(joint_positions,
                                 moving_time= move_time,
                                 accel_time=move_time/3,
                                 blocking=False)
        if(sleep_move_time == True):
            for i in range(100):
                rospy.sleep(move_time/100)
                if gui_func is not None:
                    gui_func()
        else:
            for i in range(100):
                rospy.sleep(move_time/300)
                if gui_func is not None:
                    gui_func()

    # some functions for tasks
    def move_above(self, top_face_position, angle, vertical=True):
        """!
         @brief      Sets the positions.

         @param      top_face_position  xyz in world frame
         """
        top_face_position = np.append(top_face_position,[1])
        object_position_arm = transformation_from_world_to_arm(top_face_position)
        arm_x = object_position_arm[0]
        arm_y = object_position_arm[1]
        if vertical:
            self.T = np.array([
                    [0, 0, 1, object_position_arm[0]],
                    [0, 1, 0, object_position_arm[1]],
                    [-1, 0, 0, object_position_arm[2]],
                    [0, 0, 0, 1]]) # destination
            self.T[0:3,0:3] = np.matmul(Rz(-angle), self.T[0:3,0:3])
            self.T[1,3] *= 1.04
            self.T[0,3] = self.T[0,3]*1.02 - 0.5
        else:
            theta = np.arctan2(object_position_arm[1], object_position_arm[0])
            self.T = np.array([
                [np.cos(theta), -np.sin(theta), 0, object_position_arm[0]],
                [np.sin(theta), np.cos(theta), 0, object_position_arm[1]],
                [0, 0, 1, object_position_arm[2]],
                [0, 0, 0, 1]])
        print('Arms',arm_y, arm_x)
        base_angle = np.arctan2(arm_y, arm_x)
        if vertical:
            joint_angle_guess = np.array([base_angle,0,0,-np.pi/2,0])
        else:
            joint_angle_guess = np.array([base_angle,0,0,0,0])
            
        print('joint guess', joint_angle_guess)
        temp_T = self.T.copy()
        temp_T[2,3] += 100
        print('final mid point', temp_T[:,3])
        desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.S_list, self.M_matrix, temp_T, joint_angle_guess, 0.01, 0.001,self.resp.upper_joint_limits, self.resp.lower_joint_limits)
        if IK_flag:
            self.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents, sleep_move_time=True)
            print('final mid points arrived')
        else:
            rospy.logerr("Something wrong with the IK")
        return IK_flag

    def grab(self, top_face_position, angle, is_large, vertical=True):
        if self.gripper_state:
            self.open()
        joint_angle_guess = self.get_positions()
        T_grab = self.T.copy()
        if is_large:
            T_grab[2,3] += 15
        else:
            T_grab[2,3] += 25
        
        print(T_grab[:,3])
        desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.S_list, self.M_matrix, T_grab, joint_angle_guess, 0.01, 0.001,self.resp.upper_joint_limits, self.resp.lower_joint_limits)
        # extra_torque = GravityForces(desired_joint_angle, np.array([0, 0, -9800]), self.rxarm.Mlist, self.rxarm.Glist, self.rxarm.S_list.T)
        # with open('extra_torque.txt', 'a') as outfile1:    
        #     np.savetxt(outfile1, [extra_torque], fmt='%f', delimiter= ',')
        # print('extra_torque', extra_torque)
        if IK_flag:
            self.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents)
            # actual_angle = self.get_positions()
            # angle_difference = desired_joint_angle - actual_angle
            # with open('angle_difference.txt', 'a') as outfile2:    
            #     np.savetxt(outfile2, [angle_difference], fmt='%f', delimiter= ',')
            # print('angle difference', angle_difference)
            self.close()
        else:
            rospy.logerr("Something wrong with the IK")
        return IK_flag
        

    def place_on(self, store_positions , angle, safe=False, vertical=True):
        joint_angle_guess = self.get_positions()
        T_drop = self.T.copy()
        T_drop[2,3] += 60
        print(T_drop[:,3])
        desired_joint_angle, IK_flag = IK_Base_frame_constrained(self.S_list, self.M_matrix, T_drop, joint_angle_guess, 0.01, 0.001,self.resp.upper_joint_limits, self.resp.lower_joint_limits)

        if IK_flag:
            self.set_positions_custom(desired_joint_angle, gui_func=QCoreApplication.processEvents)
            # actual_angle = self.rxarm.get_positions()
            # angle_difference = desired_joint_angle - actual_angle
            # with open('angle_difference.txt', 'a') as outfile2:    
            #     np.savetxt(outfile2, [angle_difference], fmt='%f', delimiter= ',')
            # print('angle difference', angle_difference)
            self.open()
        else:
            rospy.logerr("Something wrong with the IK")
        return IK_flag
        

    def go_to_safe(self, center=True):
        cur_pos = np.array(self.get_positions())
        upright = self.safe_position.copy()
        upright[0] = cur_pos[0]
        self.set_joint_positions(upright,
                                 moving_time=0.8,
                                 accel_time=0.4,
                                 blocking=True)
        if (center):
            self.set_joint_positions(self.safe_position,
                                    moving_time=0.3,
                                    accel_time=0.15,
                                    blocking=True)


    def reachable(self, top_face_position, vertical=True, above=True, is_large=True):
        top_face_position = np.append(top_face_position,[1])
        object_position_arm = transformation_from_world_to_arm(top_face_position)
        max_height_cone = 225
        if(np.hypot(object_position_arm[0], object_position_arm[1]) > np.hypot(250,275)*((max_height_cone - object_position_arm[2])/max_height_cone)):
            return False
        else:
            return True

    
    def stow_arm(self):
        moving_time = 1.5
        accel_time = 0.75
        self.go_to_safe(center=True)
        
        desired_pose = np.array([0, self.resp.sleep_pos[1]+(5*D2R), 0, 0, 0])
        self.publish_positions(desired_pose, moving_time, accel_time)








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
    


    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as 4*4 SE3 matrix
        """

        if len(self.S_list) == 0:
            rospy.logwarn('PoX config not loaded')
            return np.eye(4)
        cur_pos = np.array(self.get_positions())

        temp = np.eye(4)
        for i in range(5):
            temp = np.matmul(temp, expm(construct_se3_matrix(self.S_list[i,:])*cur_pos[i]))
        ee_pose = np.matmul(temp, self.M_matrix)
         
        
        return ee_pose
    


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
        self.updateEndEffectorReadout.emit(ee_transformation_to_pose(self.rxarm.get_ee_pose()).tolist())
        # for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_joint_position_pid_params(name)))
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
