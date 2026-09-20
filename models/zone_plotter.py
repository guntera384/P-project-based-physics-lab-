import numpy as np
import matplotlib.pyplot as plt
from geometry import *
#import coordinates


resolution = [1000, 1000, 1000] # (x, y, z) axis


def fresnel_radius(d1, d2, wavelength, n = 1):
    #approximation formulat for d1/d2 close to center
    D = d1 + d2
    return np.sqrt(n * wavelength * d1 * d2 / D)

def fresnel_image_bounds(A, B, wavelength, n = 1):
    Ax, Ay = A
    Bx, By = B
    L = np.linalg.norm(B - A)
    max_diameter = fresnel_radius(d1 = L/2, d2 = L/2, wavelength = wavelength, n = n)
    lower_x_bound = min(Ax, Bx) - 2 * max_diameter
    upper_x_bound = max(Ax, Bx) + 2 * max_diameter
    lower_y_bound = min(Ay, By) - 2 * max_diameter
    upper_y_bound = max(Ay, By) + 2 * max_diameter
    return np.array([lower_x_bound, upper_x_bound]), np.array([lower_y_bound, upper_y_bound])


"""def path_difference(A, B, X, Y):
    Ax, Ay = A; Bx, By = B
    L = np.linalg.norm(B - A)
    path_diff = np.sqrt((Ax - X)**2 + (Ay - Y)**2) + np.sqrt((Bx - X)**2 + (By - Y)**2) - L
    return path_diff"""


def plot_fresnel_zone(source, receiver, wavelength, n = 3):
    source = np.asarray(source)
    receiver = np.asarray(receiver)

    x_bounds, y_bounds = fresnel_image_bounds(source, receiver, wavelength, n)
    x = np.linspace(x_bounds[0], x_bounds[1], resolution[0])
    y = np.linspace(y_bounds[0], y_bounds[1], resolution[1])

    X, Y = np.meshgrid(x, y)
    path_diff = path_difference(source, receiver, X, Y)

    levels = np.arange(1, n+1) * (wavelength / 2)
    plt.contour(x, y, path_diff, levels = levels)
    plt.plot(source[0], source[1], 'o', color = 'b', label = "source")
    plt.plot(receiver[0], receiver[1], 'o', color = 'r', label = "receiver")
    plt.axis("equal")
    plt.legend()


if __name__ == "__main__":
    plot_fresnel_zone(source = [0, 1], receiver = [5, 1], wavelength = 0.02)
    plt.show()