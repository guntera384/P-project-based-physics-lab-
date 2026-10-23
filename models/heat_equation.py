import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, exp, pi, sin


def freq_fourier(X, Y, Z=[None]):
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
    else: 
        #X, Y, Z passes my meshgrid with indexing = 'ij'
        x = X[:, 0, 0]
        y = Y[0, :, 0]
        z = Z[0, 0, :]
        dx = x[1]-x[0]; dy = y[1]-y[0]; dz = z[1]-z[0] 
        kx = np.fft.fftfreq(len(x), dx)
        ky = np.fft.fftfreq(len(y), dy)
        kz = np.fft.fftfreq(len(z), dz)
        return kx, ky, kz

def fourier_spectral_gradient_2D(X, Y, func_value):
    """
    fucntion currently assumes a 2D array of data points to compute the gradient
    """
    kx, ky = freq_fourier(X, Y)
    Kx, Ky = np.meshgrid(kx, ky, indexing = 'ij')
    cf = np.fft.fft2(func_value)
    Dx_cf = 2j * pi * Kx * cf
    Dy_cf = 2j * pi * Ky * cf
    Dx = np.fft.ifft2(Dx_cf)
    Dy = np.fft.ifft2(Dy_cf)
    return Dx, Dy

def fourier_spectral_gradient_3D(X, Y, Z, func_value):
    """
    fucntion currently assumes a 3D array of data points to compute the gradient
    """
    kx, ky, kz = freq_fourier(X, Y, Z)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing = 'ij')
    cf = np.fft.fftn(func_value)
    Dx_cf = 2j * pi * Kx * cf
    Dy_cf = 2j * pi * Ky * cf
    Dz_cf = 2j * pi * Kz * cf
    Dx = np.fft.ifftn(Dx_cf)
    Dy = np.fft.ifftn(Dy_cf)
    Dz = np.fft.ifftn(Dz_cf)
    return Dx, Dy, Dz

def fourier_spectral_laplacian_2D(X, Y, func_value):
    """
    compute the laplacian with fourier interpolation
    currently limited to scalar functions
    """
    kx, ky = freq_fourier(X, Y)
    Kx, Ky = np.meshgrid(kx, ky, indexing = 'ij')
    cf = np.fft.fft2(func_value)
    D2_cf = (2j * pi) ** 2 * (Kx**2 + Ky**2) * cf
    laplacian = np.fft.ifft2(D2_cf)
    return laplacian

def fourier_spectral_laplacian_3D(X, Y, Z, func_value):
    """
    compute the laplacian with fourier interpolation
    currently limited to scalar functions
    """
    kx, ky, kz = freq_fourier(X, Y, Z)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing = 'ij')
    cf = np.fft.fftn(func_value)
    D3_cf = (2j * pi) ** 2 * (Kx**2 + Ky**2 + Kz**2) * cf
    laplacian = np.fft.ifftn(D3_cf)
    return laplacian

def make_square_symmetric(X, Y, U):
    Nx = len(X[0])  # resolution of x direction
    Ny = len(Y[:, 0])

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    Lx = Nx * dx
    Ly = Ny * dy

    x_new = X[0, 0] - Lx + dx * np.arange(2 * Nx)
    y_new = Y[0, 0] - Ly + dy * np.arange(2 * Ny)
    X_new, Y_new = np.meshgrid(x_new, y_new)

    V = np.concatenate((np.flip(U, axis=0), U), axis=0)
    V = np.concatenate((np.flip(V, axis=1), V), axis=1)
    return X_new, Y_new, V

def make_cube_symmetric(X, Y, Z, U):
    Nx = len(X[:, 0, 0])
    Ny = len(Y[0, :, 0])
    Nz = len(Z[0, 0, :])

    dx = X[1, 0, 1] - X[0, 0, 0]
    dy = Y[0, 1, 0] - Y[0, 0, 0]
    dz = Z[0, 0, 1] - Z[0, 0, 0]

def propagator_heat_equation_2D(X, Y, U, dt, alpha=1, Neumann=True):
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
        X, Y, U = make_square_symmetric(X, Y, U)
        cf0 = np.fft.fft2(U)
        kx, ky = freq_fourier(X, Y)
        Kx, Ky = np.meshgrid(kx, ky)
        cf_t = cf0 * np.exp(-alpha * 4 * pi**2 * (Kx**2 + Ky**2) * dt)
        U_t = np.fft.ifft2(cf_t)
        return U_t[Nx:, Ny:]
    if not Neumann:
        cf0 = np.fft.fft2(U)
        kx, ky = freq_fourier(X, Y)
        Kx, Ky = np.meshgrid(kx, ky)
        cf_t = cf0 * np.exp(-alpha * 4 * pi**2 * (Kx**2 + Ky**2) * dt)
        U_t = np.fft.ifft2(cf_t)
        return U_t

def propagator_heat_equation_3D(X, Y, Z, U, dt, alpha=1, Neumann=True):
    """
    input:
        X, Y: real space lattices of the function
        U: scalar function of X and Y -> U(X, Y)
        dt: timestep
        alpha: conduction coefficient (most metals around 1e-5) effects timescale
        Neumann: if True -> mirror U on x and y plane for neumann conditons (insulated square)
            if False -> make sure that U does not vary a lot on the boundary -> enlarge square
    """
    Nx = len(X[:, 0, 0])
    Ny = len(Y[0, :, 0])
    Nz = len(Z[0, 0, :])
    #not properly implemented yet
    if Neumann:
        #X, Y, Z, U = make_cube_symmetric(X, Y, U)
        cf0 = np.fft.fft2(U)
        kx, ky = freq_fourier(X, Y)
        Kx, Ky = np.meshgrid(kx, ky)
        cf_t = cf0 * np.exp(-alpha * 4 * pi**2 * (Kx**2 + Ky**2) * dt)
        U_t = np.fft.ifft2(cf_t)
        return U_t[Nx:, Ny:]
    if not Neumann:
        cf0 = np.fft.fft2(U)
        kx, ky = freq_fourier(X, Y)
        Kx, Ky = np.meshgrid(kx, ky)
        cf_t = cf0 * np.exp(-alpha * 4 * pi**2 * (Kx**2 + Ky**2) * dt)
        U_t = np.fft.ifft2(cf_t)
        return U_t

def test_plot_2D():
    alpha_copper = 1.16e-4
    alpha = 1
    resolution = 30
    x = np.linspace(0, 1, resolution, endpoint=False)
    y = np.linspace(0.1, 1.1, resolution, endpoint=False)

    X, Y = np.meshgrid(x, y)
    # U = sin(2 * pi *(X + Y)) + cos(2 * pi * X) #initial state
    U = np.ones_like(X) * 300
    U[: resolution // 4, : resolution // 4] = 400

    U0 = propagator_heat_equation(X, Y, U, 0, alpha=alpha_copper)
    U1 = propagator_heat_equation(X, Y, U, 3e2, alpha=alpha_copper, Neumann=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    mesh0 = ax[0].pcolormesh(X, Y, np.real(U0))
    mesh1 = ax[1].pcolormesh(X, Y, np.real(U1))
    plt.colorbar(mesh0, ax=ax[0])
    plt.colorbar(mesh1, ax=ax[1])
    plt.show()

def test_fourier_gradient():
    x = np.linspace(0, 1, 100, endpoint = False)
    y = np.linspace(0, 1, 80, endpoint = False)
    z = np.linspace(0, 1, 60, endpoint = False)
    X, Y, Z = np.meshgrid(x, y, z, indexing = 'ij')

    test_vals = sin(2*pi*X) * cos(4*pi*Y) * sin(6*pi*Z)
    Dx_trig, Dy_trig, Dz_trig = np.real(fourier_spectral_gradient_3D(X, Y, Z, test_vals))
    Dx_func = 2*pi*cos(2*pi*X) * cos(4*pi*Y) * sin(6*pi*Z)
    Dy_func = -4*pi*sin(2*pi*X) * sin(4*pi*Y) * sin(6*pi*Z)
    Dz_func = 6*pi*sin(2*pi*X) * cos(4*pi*Y) * cos(6*pi*Z)

    error_x = np.max(np.abs(Dx_trig - Dx_func))
    error_y = np.max(np.abs(Dy_trig - Dy_func))
    error_z = np.max(np.abs(Dz_trig - Dz_func))
    print(f"errors in x, y, z = {error_x}, {error_y}, {error_z}")

def test_fourier_spectral_laplacian_3D():
    x = np.linspace(0, 1, 100, endpoint = False)
    y = np.linspace(0, 1, 80, endpoint = False)
    z = np.linspace(0, 1, 60, endpoint = False)
    X, Y, Z = np.meshgrid(x, y, z, indexing = 'ij')

    test_vals = sin(2*pi*X) * cos(4*pi*Y) * sin(6*pi*Z)
    laplacian = -56 * pi **2 * test_vals
    laplacian_trig = np.real(fourier_spectral_laplacian_3D(X, Y, Z, test_vals))
    err = np.max(np.abs(laplacian - laplacian_trig))
    print(f"error of laplacian = {err}")

def test_make_symmetric():
    x = np.linspace(0, 1, 10, endpoint = False)
    y = np.linspace(0, 0.1, 10, endpoint = False)
    z = np.linspace(0, 0.01, 10, endpoint = False)

    X, Y, Z = np.meshgrid(x, y, z, indexing = 'ij')
    U = X + Y + Z
    #unfinished

def run_tests():
    test_fourier_gradient()
    test_fourier_spectral_laplacian_3D()
    test_make_symmetric()

if __name__ == "__main__":
    run_tests()
    """resolution = 11
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        z = np.linspace(0, 1, resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing = 'ij')
        x_test = X[:, 0, 0]
        y_test = Y[0, :, 0]
        z_test = Z[0, 0, :]
        test_func = sin(2 * pi * X) + np.ones_like(Y) + 3 * cos(4 * pi * Z)
        Dx, Dy, Dz = np.real(fourier_spectral_gradient_3D(X, Y, Z, test_func))
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(X, Y, Z, Dx, Dy, Dz, length = 0.1, normalize = True)
        plt.show()"""
