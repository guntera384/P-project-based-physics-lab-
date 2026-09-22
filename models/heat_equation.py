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

def make_square_symmetric(X, Y, U):
    Nx = len(X[0]) #resolution of x direction
    Ny = len(Y[:, 0])
    dx = X[0, 1] - X[0, 0]
    Lx = Nx * dx
    dy = Y[1, 0] - Y[0, 0]
    Ly = Ny * dy
    x_new = X[0, 0] - Lx + dx * np.arange(2 * Nx)
    y_new = Y[0, 0] - Ly + dy * np.arange(2 * Ny)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    V = np.concatenate((np.flip(U, axis=0), U), axis=0)
    V = np.concatenate((np.flip(V, axis=1), V), axis=1)
    return X_new, Y_new, V

def propagator_heat_equation(X, Y, U, dt, alpha = 1, Neumann = True):
    """
    input: 
        X, Y: real space lattices of the function
        U: scalar function of X and Y -> U(X, Y)
        dt: timestep
        alpha: conduction coefficient (most metals around 1e-5) effects timescale
        Neumann: if True -> mirror U on x and y plane for neumann conditons (insulated square)
            if False -> make sure that U does not vary a lot on the boundary -> enlarge square
    """
    Nx = len(X[0])
    Ny = len(Y[:, 0])
    if Neumann:
        #create new grid where function is mirrored
        X, Y, U = make_square_symmetric(X, Y, U)
        cf0 = np.fft.fft2(U)
        kx, ky = freq_fourier(X, Y)
        Kx, Ky = np.meshgrid(kx, ky)
        cf_t = cf0 * np.exp(-alpha * 4 * pi**2 *(Kx**2 + Ky**2) * dt)
        U_t = np.fft.ifft2(cf_t)
        return U_t[Nx:, Ny:]
    if not Neumann:
        print("not implemented yet")
        return None

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    y = np.array([0.1, 0.2, 0.3, 0.4])
    X, Y = np.meshgrid(x, y)
    U = X + Y
    X, Y, V = make_square_symmetric(X, Y, U)

    alpha_copper = 1.16e-4
    alpha = 1
    resolution = 30
    x = np.linspace(0, 1, resolution, endpoint = False)
    y = np.linspace(0.1, 1.1, resolution, endpoint = False)
        
    X, Y = np.meshgrid(x, y)
    #U = sin(2 * pi *(X + Y)) + cos(2 * pi * X) #initial state
    U = np.zeros_like(X)
    U[:resolution//4, :resolution//4] = 1
            
    U0 = propagator_heat_equation(X, Y, U, 0, alpha = alpha_copper)
    U1 = propagator_heat_equation(X, Y, U, 3e2, alpha = alpha_copper)
        
    fig, ax = plt.subplots(1, 2, figsize = (10, 4))
    mesh0 = ax[0].pcolormesh(X, Y, np.real(U0))
    mesh1 = ax[1].pcolormesh(X, Y, np.real(U1))
    plt.colorbar(mesh0, ax = ax[0])
    plt.colorbar(mesh1, ax = ax[1])
    plt.show()
