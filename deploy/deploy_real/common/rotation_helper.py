import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w

def qinverse(quat):
    # quat in w,x,y,z format
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    return np.array([w,-x,-y,-z])

def qmult(q1, q2):
    a1 = q1[0]
    b1 = q1[1]
    c1 = q1[2]
    d1 = q1[3]

    a2 = q2[0]
    b2 = q2[1]
    c2 = q2[2]
    d2 = q2[3]

    a3 = a1*a2 - b1*b2 -c1*c2 -d1*d2
    b3 = a1*b2 + b1*a2 + c1*d2 - d1*c2
    c3 = a1*c2 - b1*d2 + c1*a2 + d1*b2
    d3 = a1*d2 + b1*c2 - c1*b2 + d1*a2
    return np.array([a3, b3, c3, d3])  # w,x,y,z

def quat2mat(quat):
    # quat: w,x,y,z
    R_mat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return R_mat
