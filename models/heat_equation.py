import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, exp, pi



if __name__ == "__main__":

    alpha = 1
    resolution =11
    x = np.linspace(0, 1, resolution, endpoint = True)
    y = np.linspace(0, 1, resolution, endpoint = True)
    X, Y = np.meshgrid(x, y)
    U = sin(2 * pi *(X + Y))
    V = np.zeros(shape = (len(x), len(y)))
    V[resolution//4: resolution//2, resolution//4: resolution//2] = 1
    fourier_coeff = np.abs(np.fft.fft2(U))
    print(fourier_coeff)
    #plt.pcolormesh(X, Y, fourier_coeff)
    mesh = plt.pcolormesh(X, Y, V)
    plt.colorbar(mesh)
    plt.show()