import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, exp, pi


def freq_fourier(X, Y, Z = [None]):
    """
    function to give correct frequencies for fourier transforms
    Numpy only implements the 1D case
    """
    if Z[0] is None:
        x = X[0]
        y = Y[:, 0]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        kx = np.fft.fftfreq(len(x), dx)
        ky = np.fft.fftfreq(len(y), dy)
        return kx, ky

def fourier_spectral_gradient_2D(X, Y, func_value):
    """
    fucntion currently assumes a 2D array of data points to compute the gradient
    """
    kx, ky = freq_fourier(X, Y)
    Kx, Ky = np.meshgrid(kx, ky)
    cf = np.fft.fft2(func_value)
    Dx_cf = 2j * pi * Kx * cf
    Dy_cf = 2j * pi * Ky * cf
    Dx = np.fft.ifft2(Dx_cf)
    Dy = np.fft.ifft2(Dy_cf)
    return Dx, Dy

def fourier_spectral_laplacian_2D(X, Y, func_value):
    """
    compute the laplacian with fourier interpolation
    currently limited to scalar functions
    """
    kx, ky = freq_fourier(X, Y)
    Kx, Ky = np.meshgrid(kx, ky)
    cf = np.fft.fft2(func_value)
    D2_cf = (2j * pi)**2 *(Kx**2 + Ky**2) * cf
    laplacian = np.fft.ifft2(D2_cf)
    return laplacian

if __name__ == "__main__":
    alpha = 1
    resolution =30
    x = np.linspace(0, 1, resolution, endpoint = False)
    y = np.linspace(0, 1, resolution, endpoint = False)
    X, Y = np.meshgrid(x, y)
    U = sin(2 * pi *(X + Y))
    """V = np.zeros(shape = (len(x), len(y)))
        V[resolution//4: resolution//2, resolution//4: resolution//2] = 1
        fourier_coeff = np.fft.fft2(U)
        fig, ax = plt.subplots(1, 3, figsize = (15, 4))
        mesh0 = ax[0].pcolormesh(X, Y, U)
        mesh1 = ax[1].pcolormesh(X, Y, np.abs(fourier_coeff))
        converted = np.real(np.fft.ifft2(fourier_coeff))
        mesh2 = ax[2].pcolormesh(X, Y, converted)
        plt.colorbar(mesh0, ax = ax[0])
        plt.colorbar(mesh1, ax = ax[1])
        plt.colorbar(mesh2, ax = ax[2])"""
    fig, ax = plt.subplots(1, 2, figsize = (10, 4))
    Dx, Dy = fourier_spectral_gradient_2D(X, Y, U)
    Dx = np.real(Dx); Dy = np.real(Dy)
    mesh0 = ax[0].pcolormesh(X, Y, U)
    plt.colorbar(mesh0, ax = ax[0])
    ax[0].quiver(X, Y, Dx, Dy)
    laplace = fourier_spectral_laplacian_2D(X, Y, U)
    mesh1 = ax[1].pcolormesh(X, Y, np.real(laplace))
    plt.colorbar(mesh1, ax = ax[1])
    plt.show()