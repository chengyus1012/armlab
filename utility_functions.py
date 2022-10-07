from __future__ import print_function
from cmath import log
import imp
import numpy as np
from scipy.spatial.transform import Rotation as R
from modern_robotics import IKinBody, JacobianBody
from scipy.linalg import expm, logm, block_diag

# from spatialmath import *

def Rx(a):
    return np.array([[1, 0, 0],  
                     [0, np.cos(a), -np.sin(a)],  
                     [0, np.sin(a), np.cos(a)]])

def Ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)],  
                     [0, 1, 0],  
                     [-np.sin(a), 0, np.cos(a)]])

def Rz(a): 
    return np.array([[np.cos(a), -np.sin(a), 0],  
                     [np.sin(a), np.cos(a), 0],  
                     [0, 0, 1]])

def construct_se3_matrix(vector):
    """!
    @brief      Utility function for constructing se3 matrix from a 6d vector.

    @return     4*4 se3 matrix
    """
    w = np.zeros((3,3))
    w[0,1] = -vector[2]
    w[1,0] = vector[2]
    w[0,2] = vector[1]
    w[2,0] = -vector[1]
    w[1,2] = -vector[0]
    w[2,1] = vector[0]
    v = vector[3:].reshape((3,1))
    S = np.block([[w,v],[np.zeros((1,4))]])
    # print("SE3:", vector, S,sep="\n")
    return S

def construct_SI_mat(inertia_vector, mass):
    """!
    @brief      Utility function for constructing spatial inertia matrix.

    @return     6*6 matrix
    """
    inertia_mat = np.array([
        [inertia_vector[0], inertia_vector[3], inertia_vector[4]],
        [inertia_vector[3], inertia_vector[1], inertia_vector[5]],
        [inertia_vector[4], inertia_vector[5], inertia_vector[2]]
    ])
    mass_mat = mass*np.eye(3)
    return block_diag(inertia_mat, mass_mat)




def InvOfTrans(T):
    """!
    @brief      Calculate the inverse of transformation matrix
    @param T    4*4 transformation matrix
    @return     Inverse of transformation matrix
    """
    T = np.array(T)
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    Rt = np.array(R).T

    return np.block([[Rt, -np.matmul(Rt, p).reshape((3,1))],[0, 0, 0, 1]])

def conv_se3_vec(se3mat):
    return np.array([se3mat[2][1], se3mat[0][2], se3mat[1][0],
                 se3mat[0][3], se3mat[1][3], se3mat[2][3]]).reshape((6,1))

def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])


def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    T = np.array(T)
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(VecToso3(p), R), R]]

def write_to_file(path, array):
        with open(path, 'w') as outfile:
            for slice_2d  in array:
                np.savetxt(outfile, slice_2d, fmt='%f', delimiter= ',')

def to_rotation(q):
    """
    Convert a quaternion to the corresponding rotation matrix.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    q = q / np.linalg.norm(q)
    vec = q[:3]
    w = q[3]

    R = (2*w*w-1)*np.identity(3) - 2*w*VecToso3(vec) + 2*vec[:, None]*vec
    return R

def to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0,0] - R[1,1] - R[2,2]
            q = [t, R[0, 1]+R[1, 0], R[2, 0]+R[0, 2], R[1, 2]-R[2, 1]]
        else:
            t = 1 - R[0,0] + R[1,1] - R[2,2]
            q = [R[0, 1]+R[1, 0], t, R[2, 1]+R[1, 2], R[2, 0]-R[0, 2]]
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0,0] - R[1,1] + R[2,2]
            q = [R[0, 2]+R[2, 0], R[2, 1]+R[1, 2], t, R[0, 1]-R[1, 0]]
        else:
            t = 1 + R[0,0] + R[1,1] + R[2,2]
            q = [R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0], t]

    q = np.array(q) # * 0.5 / np.sqrt(t)
    return q / np.linalg.norm(q)

def quaternion_normalize(q):
    """
    Normalize the given quaternion to unit quaternion.
    """
    return q / np.linalg.norm(q)

def ee_transformation_to_pose(T):
    """
    Convert end effector transformation matrix to end effector pose with Euler angles
    """

    pos = T[:3,3]
    rot = T[:3,:3]

    r = R.from_dcm(rot)
    euler = r.as_euler('zyx')
    return np.concatenate([pos,euler])

def transformation_from_arm_to_world(T):
    """!
    @brief      transform a pose from arm to_world.

    @return     4*4 SE3 matrix representing pose in world
    """
    T_arm_to_world = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    pose_world = np.matmul(T_arm_to_world, T)
    return  pose_world


def transformation_from_world_to_arm(T):
    """!
    @brief      transform a pose from arm to_world.

    @return     4*4 SE3 matrix representing pose in world
    """
    T_world_to_arm = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    pose_arm = np.matmul(T_world_to_arm, T)
    return  pose_arm


class Block:
    # Size "Enum"
    SMALL = 0
    LARGE = 1

    LARGE_BLOCK_THRESHOLD = 750

    SMALL_MM = 24.0 # Actual size in mm
    LARGE_MM = 38.1 # Actual size in mm

    def __init__(self, top_face_position, angle, is_large, ignore=False, contour=None, color=None, possible_blocks_beneath = []):
        self.top_face_position = top_face_position # Defined to center of top face
        self.angle = angle
        self.is_large = is_large
        self.ignore = ignore
        self.color = color
        self.possible_blocks_beneath = possible_blocks_beneath

    def __str__(self):
        return "XYZ: ({0:.2f}, {1:.2f}, {2:.2f})  Angle: {3:.2f}  Large: {4}  Color: {5}".format(self.top_face_position[0], self.top_face_position[1], self.top_face_position[2], self.angle, self.is_large, self.color)
    # def is_same(self, other):


    
