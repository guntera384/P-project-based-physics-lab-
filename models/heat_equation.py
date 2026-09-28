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
    resolution = 30
    x = np.linspace(0, 1, resolution, endpoint = False)
    y = np.linspace(0, 1, resolution, endpoint = False)
    X, Y = np.meshgrid(x, y)
    U = sin(2 * pi *(X + Y)) + cos(2 * pi * X) #initial state
    fig, ax = plt.subplots(1, 2, figsize = (10, 4))
    cf0 = np.fft.fft2(U)
    kx, ky = freq_fourier(X, Y)
    Kx, Ky = np.meshgrid(kx, ky)
    cf = lambda t: cf0 * np.exp(-alpha * 4 * pi**2 *(Kx**2 + Ky**2) * t)
    U0 = np.fft.ifft2(cf(0))
    #U1 = np.fft.ifft2(cf(2.5e-2))
    U1 = np.fft.ifft2(cf(1))
    mesh0 = ax[0].pcolormesh(X, Y, np.real(U0))
    mesh1 = ax[1].pcolormesh(X, Y, np.real(U1))
    plt.colorbar(mesh0, ax = ax[0])
    plt.colorbar(mesh1, ax = ax[1])
    plt.show()
