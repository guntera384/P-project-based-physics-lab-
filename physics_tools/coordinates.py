import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, e, exp, pi, sin

resolution = [1000, 1000, 1000]  # (x, y, z) axis

x_basis = np.arange(0, resolution[0] + 1)  # rename needed
y_basis = np.arange(0, resolution[1] + 1)
z_basis = np.arange(0, resolution[2] + 1)
origin = np.array([0, 0])
dimension = 2  # perhaps adjustable later, currently unused

cartesian_min_x = 0  # size adjustment not properly implemented yet
cartesian_max_x = 10
assert cartesian_min_x < cartesian_max_x

cartesian_min_y = 0
cartesian_max_y = 10
assert cartesian_min_y < cartesian_max_y

cartesian_min_z = 0
cartesian_max_z = 10
assert cartesian_min_z < cartesian_max_z

source_1 = np.array([2, 5], dtype=float)  # source vector
source_2 = np.array([8, 5], dtype=float)

sources = [source_1, source_2]


def scale_factors_xyz(dim=3) -> np.ndarray:
    x_res = resolution[0]
    y_res = resolution[1]
    z_res = resolution[2]
    cart_range_x = cartesian_max_x - cartesian_min_x
    cart_range_y = cartesian_max_y - cartesian_min_y
    cart_range_z = cartesian_max_z - cartesian_min_z
    x_scale = x_res / cart_range_x
    y_scale = y_res / cart_range_y
    z_scale = z_res / cart_range_z
    if dim == 2:
        return [x_scale, y_scale]
    if dim == 3:
        return [x_scale, y_scale, z_scale]


def cartesian_point_to_pixel(cart: np.ndarray) -> np.ndarray:
    assert len(cart) == 3 or len(cart) == 2
    assert cartesian_min_x <= cart[0] <= cartesian_max_x
    # print(cartesian_min_y, cart[1], cartesian_max_y)
    assert cartesian_min_y <= cart[1] <= cartesian_max_y
    x = cart[0]
    y = cart[1]
    metric = scale_factors_xyz()
    x_pixel = x * metric[0]
    y_pixel = y * metric[1]
    if len(cart) == 2:
        return np.array([x_pixel, y_pixel])
    if len(cart) == 3:
        assert cartesian_min_z <= cart[2] <= cartesian_max_z
        z = cart[2]
        z_pixel = z * metric[2]
        return np.array([x_pixel, y_pixel, z_pixel])


def deg_to_rad(deg):
    return deg / 360 * 2 * pi


def rad_to_deg(rad):
    return rad / (2 * pi) * 360


def polar_to_cartesian(polar, angle="rad"):
    r = polar[0]
    theta = polar[1]
    if angle == "deg":
        theta = deg_to_rad(theta)
    return np.array([r * cos(theta), r * sin(theta)])


def double_pendulum_conv(
    w_p: float,
    h_p: float,
    w_f: float = 1.03,
    h_f: float = 1.1,
    h_j: float = 1.3,
    l_1: float = 47.5,
):
    """
    Converts Philipp's measurement format to an input that double_pendulum_angles can read, i.e. distance to top joint, cf. double_pendulum_angles' docstring.
    w_p is the horizontal distance from the right edge of the left rail to the left edge of the rod holder.
    h_p is the vertical distance from the pendulum suspension bar to the rod holder bar,
    w_f is the half-width of the rod holder, h_f is the height of the rod holder,
    h_j is the height of the top joint holder, l_1 is the half-width of the rack,
    i.e. the horizontal distance from the right edge of the left bar to the top joint.
    """
    return (w_p + w_f - l_1), (h_p + h_f - h_j)


def double_pendulum_inv(
    x: float,
    y: float,
    w_f: float = 1.03,
    h_f: float = 1.1,
    h_j: float = 1.3,
    l_1: float = 47.5,
):
    """
    Inverts double_pendulum_conv, i.e. turns x,y into w_p, h_p
    """
    return (x - w_f + l_1), (y - h_f + h_j)


def double_pendulum_angles(
    x: float, y: float, right: bool = False, l1: float = 20, l2: float = 15
):
    """
    Converts an x, y position to angles in the double pendulum.
    Given:
    x, y coordinates measured right and up from the top joint of the double pendulum;
    right indicating the position of the middle joint
    (relative to the line connecting the top joint to the mass),
    l1, l2 the lengths of the corresponding pendulum arms.
    All lengths must be given in cm.
    Return:
    ang1, ang2 of the top and bottom joint relative to the neutral position (equilibrium).
    Formulae given by derivation in calcxy.pdf
    Given: (0, l1-l2) expect: 0, pi
    Given: (0, l1+l2) expect: 0, 0
    """
    char = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    delta = math.acos(char)
    c = x**2 + y**2
    phi = math.atan(l2 * math.sin(delta) / (l1 + l2 * char))
    theta2 = math.asin(x / c) - phi
    theta1 = theta2 + delta
    return theta1, theta2


def double_pendulum_xy(theta_1: float, theta_2: float, l1: float = 20, l2: float = 15):
    """
    Inverts double_pendulum_angles, i.e. converts two angles into (x,y) coordinates
    as given in the above description.
    """
    return l1 * sin(theta_1) + l2 * sin(theta_2), -l1 * cos(theta_1) - l2 * cos(theta_2)


if __name__ == "__main__":
    polar = [3, 90]
    cartesian = polar_to_cartesian(polar, angle="deg")
    print(cartesian)
