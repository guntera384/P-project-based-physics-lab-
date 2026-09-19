import numpy as np
import matplotlib.pyplot as plt
from coordinates import *

if __name__ == "__main__":
    a = [1, 0]
    b = [0.5, 1]
    n_cells_a = 20
    n_cells_b = 20
    principal_vectors = [a, b]
    n1 = np.arange(-n_cells_a //2, n_cells_a//2 + 1)
    n2 = np.arange(-n_cells_b //2, n_cells_b//2 + 1)
    N1, N2 = np.meshgrid(n1, n2)
    X = N1 * a[0] + N2 * b[0]
    Y = N1 * a[1] + N2 * b[1]
    plt.scatter(X, Y, color = 'k')
    plt.plot(0, 0, 'o', markersize = 12, color = 'b')
    plt.quiver(0, 0, a[0], a[1], color = 'r', label = 'a vector', angles = 'xy', scale_units = 'xy', scale = 1)
    plt.quiver(0, 0, b[0], b[1], color = 'g', label = 'b vector', angles = 'xy', scale_units = 'xy', scale = 1)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    plt.show()