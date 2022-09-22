from __future__ import print_function
import numpy as np
from scipy.spatial.transform import Rotation as R
# from spatialmath import *


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

def write_to_file(path, array):
        with open(path, 'w') as outfile:
            for slice_2d  in array:
                np.savetxt(outfile, slice_2d, fmt='%f', delimiter= ',')

def skew(vec):
    """
    Create a skew-symmetric matrix from a 3-element vector.
    """
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])

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

    R = (2*w*w-1)*np.identity(3) - 2*w*skew(vec) + 2*vec[:, None]*vec
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


# A = np.eye(4,4)
# x = SE3(A)
# x.animate(frame = 'A')