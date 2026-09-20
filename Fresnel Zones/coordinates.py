import numpy as np
from numpy import sin, cos, pi, e, exp
import matplotlib.pyplot as plt


resolution = [1000, 1000, 1000] # (x, y, z) axis

x_basis = np.arange(0, resolution[0] + 1) #rename needed
y_basis = np.arange(0, resolution[1] + 1)
z_basis = np.arange(0, resolution[2] + 1)
origin = np.array([0, 0])
dimension = 2 #perhaps adjustable later, currently unused

cartesian_min_x = 0 #size adjustment not properly implemented yet
cartesian_max_x = 10
assert(cartesian_min_x < cartesian_max_x)

cartesian_min_y = 0
cartesian_max_y = 10 
assert(cartesian_min_y < cartesian_max_y)

cartesian_min_z = 0
cartesian_max_z = 10
assert(cartesian_min_z < cartesian_max_z)

source_1 = np.array([2, 5], dtype = float) # source vector
source_2 = np.array([8, 5], dtype = float)

sources = [source_1, source_2]

def scale_factors_xyz(dim = 3) -> np.ndarray:
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
        return [x_scale, y_scale, z_scale ]



def cartesian_point_to_pixel(cart: np.ndarray) -> np.ndarray:
    assert(len(cart) == 3 or len(cart) == 2)
    assert(cartesian_min_x <= cart[0] <= cartesian_max_x)
    #print(cartesian_min_y, cart[1], cartesian_max_y)
    assert(cartesian_min_y <= cart[1] <= cartesian_max_y)
    x = cart[0]
    y = cart[1]
    metric = scale_factors_xyz()
    x_pixel = x* metric[0]
    y_pixel = y * metric[1]
    if len(cart) == 2:
        return np.array([x_pixel, y_pixel])
    if len(cart) == 3:
        assert(cartesian_min_z <= cart[2] <= cartesian_max_z)
        z = cart[2]
        z_pixel = z * metric[2]
        return np.array([x_pixel, y_pixel, z_pixel])

def deg_to_rad(deg):
    return deg / 360 * 2 * pi

def rad_to_deg(rad):
    return rad / (2 * pi) * 360

def polar_to_cartesian(polar, angle = "rad"):
    r = polar[0]
    theta = polar[1]
    if angle == "deg":
        theta = deg_to_rad(theta)
    return np.array([r* cos(theta), r * sin(theta)])


if __name__ == "__main__":
    polar = [3, 90]
    cartesian = polar_to_cartesian(polar, angle = "deg")
    print(cartesian)

