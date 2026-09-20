import numpy as np
import matplotlib.pyplot as plt
from coordinates import *


def build_lattice(a, b, c = [None], n_cells = None):
    """
    input: 
        a, b, c: primitive lattice vectors
        n_cells: number of cells in any direction
    output: 
        X, Y, (Z) meshgrid of points
        Z is only returned if c is provided
    """
    
    if c[0] == None:
        if n_cells == None:
            n_cells = 20 #needed as 2D and 3D plotting require different default sizes
        n1 = np.arange(-n_cells //2, n_cells//2 + 1)
        n2 = np.arange(-n_cells //2, n_cells//2 + 1)
        N1, N2 = np.meshgrid(n1, n2)
        X = N1 * a[0] + N2 * b[0]
        Y = N1 * a[1] + N2 * b[1]
        return X, Y
    else:
        assert(len(a) == 3 and len(b) == 3 and len(c) == 3)
        if n_cells == None:
            n_cells = 5
        n1 = np.arange(-n_cells //2, n_cells//2 + 1)
        n2 = np.arange(-n_cells //2, n_cells//2 + 1)
        n3 = np.arange(-n_cells//2, n_cells//2 + 1)
        N1, N2, N3 = np.meshgrid(n1, n2, n3)
        X = N1 * a[0] + N2 * b[0] + N3 * c[0]
        Y = N1 * a[1] + N2 * b[1] + N3 * c[1]
        Z = N1 * a[2] + N2 * b[2] + N3 * c[2]
        return X, Y, Z


def plot_lattice(X, Y, Z = [None]):
    fig = plt.figure()
    if Z[0] == None:
        plt.scatter(X, Y, color = 'k')
        plt.plot(0, 0, 'o', markersize = 12, color = 'b')
        plt.quiver(0, 0, a[0], a[1], color = 'r', label = 'a vector', angles = 'xy', scale_units = 'xy', scale = 1)
        plt.quiver(0, 0, b[0], b[1], color = 'g', label = 'b vector', angles = 'xy', scale_units = 'xy', scale = 1)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.legend()
    else:
        ax = fig.add_subplot(projection = "3d")
        ax.scatter(X, Y, Z, color = 'k')
        ax.plot(0, 0, 0, 'o', markersize = 12, color = 'y')
        ax.quiver(0, 0, 0, a[0], a[1], a[2], color = 'r', label = 'a') #, color = 'r', label = 'a vector', angles = 'xy', scale_units = 'xy', scale = 1
        ax.quiver(0, 0, 0, b[0], b[1], b[2], color = 'g', label = 'b vector')
        ax.quiver(0, 0, 0, c[0], c[1], c[2], color = 'b', label = 'b vector')
        plt.legend(loc = "upper left")
    plt.show() 

def create_principal_vectors(type = "square", lengths = np.array([1, 1, 1]), angles = np.array([90, 90, 90])):
    """
    input: 
        type: crytal type to be generated
        lengths: lengths of the vectors a, b, c
        angles: alpha, beta, gamma of the unit cell
    output: a, b, (c) principal vectors for the common lattice types
    """
    if type == "square":
        a = np.array([lengths[0], 0])
        b = np.array([0, lengths[1]])
        return a, b




if __name__ == "__main__":
    a, b = create_principal_vectors()
    X, Y = build_lattice(a, b)
    plot_lattice(X, Y)
    