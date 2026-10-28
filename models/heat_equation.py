import matplotlib.pyplot as plt
import scipy as sp
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

    U0 = propagator_heat_equation_2D(X, Y, U, 0, alpha=alpha_copper)
    U1 = propagator_heat_equation_2D(X, Y, U, 3e2, alpha=alpha_copper, Neumann=True)

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

def make_T_initial(x, lengthscale, T0, delta_T):
    T = T0 + delta_T * exp(-x/lengthscale)
    return T

def decay_time_mode_heat_eq(L, n, alpha):
    #gives the time for any mode to decay from A_0 to A_0 * 1/e
    return L**2 / ((n * pi)**2 * alpha)

def decay_time_heat_eq(n = 1, timescale = "minutes"):
    """
    describes how long the system needs to decay for different configurations
    n: mode considered for decay time
    timescale: gives timescale of result -> minutes or seconds
    """
    materials = {"copper": 1.16e-4, 
                 "aluminium": 0.97e-4,
                 "brass": 0.34e-4,
                 "iron": 0.23e-4,
                 "stainless steel": 0.04e-4}
    
    lengths = [0.1, 0.2, 0.3]
    if timescale == "minutes":
        time_factor = 60

    if timescale == "seconds":
        time_factor = 1

    for material in materials: 
        alpha = materials[material]
        mat = np.zeros(len(lengths))
        for i,l in enumerate(lengths):
            mat[i] = decay_time_mode_heat_eq(l, n, alpha) / time_factor
        print(f"decay times {material}: {mat} in {timescale}")

def evaluate_point_heat_sum(coeff, t, index, alpha, L): #renaming needed
    N = len(coeff)
    delta_x = N/L
    T = coeff[0]
    for n in range(1,N):
        T += exp(- alpha (pi / L)**2 * n **2 *t) * cos(n * (pi/L) * (index + 0.5) * delta_x)
    return T
    
def evaluate_cosine_sum(coeff, x, L):
    out = 0
    for n in np.arange(len(coeff)):
        out += coeff[n] * cos(n * (pi /L) * x)
    return out

def lossless_propagation_1D(cos_coeffs, t, L = 0.3, alpha = 0.34e-4):
    """
    takes initial measurement data and finds the later state assuming no losses
    input:
        cos_coeffs: coefficients of starting temperature in cosine basis
        pos_sensor: positions of sensor corresponding to T_initial 
            pos_sensor[0] is location of T_initial[0]
    """
    cos_coeffs_propagated = np.zeros_like(cos_coeffs)
    cos_coeffs_propagated[0] = cos_coeffs[0]
    for i in range(1, len(cos_coeffs)):
        cos_coeffs_propagated[i] = cos_coeffs[i] * exp(- alpha *(pi * i /L)**2 * t)
    return cos_coeffs_propagated

def stored_heat_rod(l, d, delta_T, Cv = 449, density = 7874):
    """
    inputs: 
        l: length of the rod [m]
        d: diameter of the rod [m]
        delta_T: temperature difference considered (T - T_0)
        Cv: specific heat capacity per weight [J/kg K] assumed constant
        density: density of material [kg/m^3]
    output:
        U: Difference in internal energy [Joules]
    """
    V = pi * d**2 / 4 * l
    M = V * density
    delta_U = Cv * delta_T * M
    return delta_U

def power_radiated_rod(l, d, epsilon, T_sys, T_env = 293.15):
    """
    assumed constant temperature distribution as a rough upper bound
    inputs: 
        l: length of the rod [m]
        d: diameter of the rod [m]
        epsilon: emissivity
        T_sys: Temperature of the system considered (uniform over surface)
        T_env: Temperature of the surrounding environment (T = 293.15 K by default / roomtemp)
    out: 
        P_rad: Power radiated per second [J/s]
    """
    A = pi * d * l
    sigma_B = 5.67e-8
    P_rad = A * sigma_B * epsilon * (T_sys**4 - T_env**4)
    return P_rad

def temperature_difference_from_delta_U(l, d, delta_U, Cv = 449, density = 7874):
    V = pi * d**2 / 4 * l
    M = V * density
    delta_T = delta_U /(Cv * M)
    return delta_T

def estimates_temperature_loss():
    """
    crude model to keep code. Should be adjust later
    """
    l = 0.3
    d = 0.03
    T_env = 293.15
    T_sys = 353.15
    delta_T = T_sys - T_env
    epsilon = 0.3
    delta_U = stored_heat_rod(l, d, delta_T)
    print(f"delta_U = {delta_U}J")
    P_rad = power_radiated_rod(l, d, epsilon, T_sys, T_env)
    print(f"Power radiated = {P_rad} [J/s]")
    delta_U = delta_U - 4 * P_rad * 600
    T_final = temperature_difference_from_delta_U(l, d, delta_U) + T_env
    print(f"final temperature = {T_final}")

def get_coeffs_cos_basis(T, x):
    """
    find the coefficients to expand the array T in a cos basis
    """
    N = len(x)
    L = (x[1] - x[0]) * N
    coeffs = np.zeros_like(T)
    coeffs[0] = 1/N * np.sum(T)
    for n in range(1, N):
        coeffs[n] = 2/N * np.sum(T * cos(n * pi /L * x))
    return coeffs

if __name__ == "__main__":
    N = 6
    L = 0.3
    T_initial = np.array([360, 300, 300, 300, 300, 300], dtype = float) #temperature values measured at nodes
    delta_x = L / N
    x = delta_x * np.arange(N) + (delta_x/2) #get a vector with the positions implemented
    coeffs = get_coeffs_cos_basis(T_initial, x)
    print(coeffs)
    out = evaluate_cosine_sum(coeffs, x, L)
    print(out)
    


    