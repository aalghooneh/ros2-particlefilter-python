

import numpy as np
from sensor_msgs.msg import LaserScan
from math import atan2, asin


def euler_from_quaternion(quat):
    """
    Convert quaternion (w in last place) to euler roll, pitch, yaw.
    quat = [x, y, z, w]
    """
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)
    # just unpack yaw for tb
    return yaw

def position_2_cell(cartesianPoints: np.array, origin: np.array, res):
    return (cartesianPoints - origin)/res


def cell_2_position(cells, origin: np.array, res):
    return np.floor(cells * res + origin)


def convertScanToCartesian(laserScan: LaserScan):

    angle_min = laserScan.angle_min
    angle_increment = laserScan.angle_increment
    range_min = laserScan.range_min
    range_max = laserScan.range_max
    ranges = np.array(laserScan.ranges)

    valid_indices = np.where((ranges != 0) & (ranges <= range_max)) & (ranges >= range_min)
    valid_ranges = ranges[valid_indices]

    angles = angle_min + valid_indices[0] * angle_increment

    cartesian_points = np.column_stack((valid_ranges * np.cos(angles), valid_ranges * np.sin(angles)))
    cartesian_points_homo = np.column_stack((cartesian_points, np.ones(cartesian_points.shape[0])))

    return cartesian_points, cartesian_points_homo