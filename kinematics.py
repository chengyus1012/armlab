"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""
from __future__ import print_function
import imp
import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from utility_functions import *

def Jacobian_Baseframe(S, jointangles):
    """!
    @brief      Computes the fixed base frame Jacobian
    @param S    The joint screw axes in the base frame when the
                    manipulator is at the home position
    @param jointangles A list of joint coordinates
    @return     Inverse of transformation matrix
    """
    Js = np.array(S).T.copy().astype(np.float)
    T = np.eye(4)
    for i in range(1, len(jointangles)):
        T = np.matmul(T, expm(construct_se3_matrix(np.array(S)[i - 1,:] \
                                * jointangles[i - 1])))
        # print('Shapes:', Adjoint(T).shape, np.array(S)[i, :].shape)
        Js[:, i] = np.squeeze(np.matmul(Adjoint(T), np.array(S)[i, :].reshape((6,1))))
    return Js

def FK_Baseframe(joint_angle_dis, M, S_list):
    """!
    @brief      FK in body frame.

    @return     4*4 SE3 matrix representing ee pose in body frame
    """

    temp = np.eye(4)
    print(joint_angle_dis)
    for i in range(5):
        # print('More shapes',i,construct_se3_matrix(S_list[i,:]).shape,end=', ')
        # print(construct_se3_matrix(S_list[i,:]).shape,joint_angle_dis[i])
        # print((construct_se3_matrix(S_list[i,:])*joint_angle_dis[i]).shape,end=", ")
        # print(expm(construct_se3_matrix(S_list[i,:])*joint_angle_dis[i]).shape)
        temp = np.matmul(temp, expm(construct_se3_matrix(S_list[i,:])*joint_angle_dis[i]))
    ee_pose = np.matmul(temp, M)
    
    return ee_pose



def IK_Base_frame(S, M, T, joint_angles_guess, e_w, e_v):
    """!
    @brief      Computes inverse kinematics in the fixed base frame
    @param S    The joint screw axes in the base frame when the
                    manipulator is at the home position
    @param M    Home position for the end effector
    @param T    The desired end effector position
    @param jointangles A list of joint coordinates
    @param e_w  A small positive tolerance on the end-effector orientation error
    @param e_v  A small positive tolerance on the end-effector linear position error
    @return     Inverse of transformation matrix
    """


    joint_angles = np.array(joint_angles_guess).copy()
    i = 0
    maxiterations = 20

    cur_pose = FK_Baseframe(joint_angles, M, S)


    error_SE3_b = np.matmul(InvOfTrans(cur_pose), T)
    vector_twist_b = conv_se3_vec(logm(error_SE3_b))
    vector_twist_s = np.matmul(Adjoint(cur_pose), vector_twist_b)

    err = np.linalg.norm([vector_twist_s[0], vector_twist_s[1], vector_twist_s[2]]) > e_w \
          or np.linalg.norm([vector_twist_s[3], vector_twist_s[4], vector_twist_s[5]]) > e_v

    while err and i<maxiterations:
        joint_angles = joint_angles + np.squeeze(np.matmul(np.linalg.pinv(Jacobian_Baseframe(S, joint_angles)), vector_twist_s.reshape((6,1))))
        
        i+=1

        cur_pose = FK_Baseframe(joint_angles, M, S)
        error_SE3_b = np.matmul(InvOfTrans(cur_pose), T)
        vector_twist_b = conv_se3_vec(logm(error_SE3_b))
        vector_twist_s = np.matmul(Adjoint(cur_pose), vector_twist_b)

        print('Errs',np.linalg.norm([vector_twist_s[0], vector_twist_s[1], vector_twist_s[2]]), \
          np.linalg.norm([vector_twist_s[3], vector_twist_s[4], vector_twist_s[5]]))
        err = np.linalg.norm([vector_twist_s[0], vector_twist_s[1], vector_twist_s[2]]) > e_w \
          or np.linalg.norm([vector_twist_s[3], vector_twist_s[4], vector_twist_s[5]]) > e_v
    return (joint_angles, not err)

def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    pass


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    pass


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    pass


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass