# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -jupyter
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#   kernelspec:
#     display_name: .venv (3.13.14)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This serves to give estimates for the experiment heat equation.
# I will try to keep it as general as possible to adjust quickly. Functions are implemented in heat_equation.py
# Units are chosen in si. Use meters as in cm the diffusivity values change the order of magnitude a lot !

# %%
import numpy as np
import matplotlib.pyplot as plt
from heat_equation import *

# %% [markdown]
# First we take the scenario of a lossless rod.
# The following block takes an array of measurement points at time $t = 0$ and finds the solution at a later time $t = t_{final}$
# Example setup for the block at roughly 80C and rod at room temperature

# %%
#initial_state = [360, 300, 300, 300, 300, 300] #temperature values measured at nodes
N = 6
L = 0.3
d = 0.025 #diameter of rod
alpha = 0.34e-4 #conductivity of brass
t_final = 300
#coeffs = np.array([300, 20, 20, 20, 20, 20], dtype = float)
T_initial = np.array([360, 300, 300, 300, 300, 300], dtype = float) #temperature values measured at nodes
delta_x = L / N
x = delta_x * np.arange(N) + (delta_x/2) #get a vector with the positions implemented
coeffs = get_coeffs_cos_basis(T_initial, x)
coeffs_50 = lossless_propagation_1D(coeffs, t = 50) #propagate coeffs in time
coeffs_300 = lossless_propagation_1D(coeffs, t = 300) #propagate coeffs in time
coeffs_1200 = lossless_propagation_1D(coeffs, t = 1200) #propagate coeffs in time
print(f"intial cos basis coefficients: {np.round(coeffs, 2)}")
print(f"t = 50 cos basis coefficients: {np.round(coeffs_50, 3)}") 
print(f"t = 300 cos basis coefficients: {np.round(coeffs_300, 3)}")
print(f"t = 1200 cos basis coefficients: {np.round(coeffs_1200, 3)}")

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
ax[0,0].scatter(x, T_initial)
ax[0,0].set(ylim = (295, 365))
ax[0,0].set_xlabel("position on rod [m]")
ax[0,0].set_ylabel("Temperature at sensor [K]")
ax[0,0].set_title(f"Temperature distribution at t = {0}")

ax[0,1].scatter(x, evaluate_cosine_sum(coeffs_50, x, L = L))
ax[0,1].set(ylim = (295, 365))
ax[0,1].set_xlabel("position on rod [m]")
ax[0,1].set_ylabel("Temperature at sensor [K]")
ax[0,1].set_title(f"Temperature distribution at t = {50}")

ax[1,0].scatter(x, evaluate_cosine_sum(coeffs_300, x, L = L))
ax[1,0].set(ylim = (295, 365))
ax[1,0].set_xlabel("position on rod [m]")
ax[1,0].set_ylabel("Temperature at sensor [K]")
ax[1,0].set_title(f"Temperature distribution at t = {300}")

ax[1,1].scatter(x, evaluate_cosine_sum(coeffs_1200, x, L = L))
ax[1,1].set(ylim = (295, 365))
ax[1,1].set_xlabel("position on rod [m]")
ax[1,1].set_ylabel("Temperature at sensor [K]")
ax[1,1].set_title(f"Temperature distribution at t = {1200}")
plt.show()

# %% [markdown]
# As you can see we need to make the measurement rather quickly to not have the earliest modes decay a lot. 

# %%
#time for system to relax to constant temperature distribution
r_tol = 0.01
relaxation_time = -(L/pi)**2 /alpha * np.log(r_tol)
print(f"time to reach constant temperature = {relaxation_time}")

# %% [markdown]
# This segment serves to estimate the losses in our system. All estimates are rather crude to find how significant they are. The rest needs to be determined experimentally.

# %%
#loss through warming of air
V_casing = 0.0525**2 * 0.35 #(5.25cm)^2 * 35cm volume of enclosure
V_rod = L * (pi * (d/2)**2)
A_rod = L * (pi * d)
V_air = V_casing - V_rod
print("---system parameters---")
print(f"Volume of  of air inside casing / rod = {(V_air) * 1e3} e-3 m^3")
print(f"Volume of rod = {V_rod * 1e3} e-3 m^3")
print(f"Area of rod: {A_rod} m^2")

C_p_brass = 380 #J/kg/K
rho_brass = 8850 #kg / m^3
M_rod = rho_brass * V_rod

mol_per_m3 = 41 #moles per m^3 of air
n_air = V_air  * mol_per_m3 #number of moles
print(f"numer of moles of air = {n_air}")
R = 8.31 #J /K / mol
gamma = 1.4 #diatomic ideal gas
C_p_air = gamma * (n_air * R) / (gamma - 1)

#heating all air to maximum temperature (20C -> 80C)
delta_T_air = 60
delta_U_air = C_p_air * delta_T_air
delta_T_brass = delta_U_air /(M_rod * C_p_brass)
print("---losses of the rod---")
print(f"change in rod temperature through heat exchange with air: {delta_T_brass}")



# %%
#losses via black body radiation (estimate without insulation)
fraction = 0.3 #fraction of rod treated at max temperature
emissivity = 0.3
#treat the problem in 3 time intervals: 0-50s, 50-300s, 300-1200s

#0-50s hot part constant at T_max, cold part not yet warmed
P1_high = power_radiated_rod(L * fraction, d, epsilon = emissivity, T_sys = 353.15)
P1_low = power_radiated_rod(L * (1-fraction), d, epsilon = emissivity, T_sys = 293.15)
delta_U_rad_1 = 50 * (P1_high + P1_low)
delta_T_brass_rad1 = delta_U_rad_1 /(M_rod * C_p_brass)
delta_T_brass_rad1_partial = delta_U_rad_1 /(fraction * M_rod * C_p_brass)
print("---heat lost during different intervals---")
print(f"0-50s interval (heat lost on total bar): {delta_T_brass_rad1}")
print(f"0-50s interval (heat lost on hot zone): {delta_T_brass_rad1_partial}")

#50-300s hot part at 323.15, cold part at 298.15K
P2_high = power_radiated_rod(L * fraction, d, epsilon = emissivity, T_sys = 323.15)
P2_low = power_radiated_rod(L * (1-fraction), d, epsilon = emissivity, T_sys = 298.15)
delta_U_rad_2_hot = 250 * P2_high
delta_U_rad_2_cold = 250 * P2_low
delta_U_rad_2 = delta_U_rad_2_hot + delta_U_rad_2_cold
delta_T_brass_rad2 = delta_U_rad_2 /(M_rod * C_p_brass)
delta_T_brass_rad2_hot = delta_U_rad_2_hot /(fraction * M_rod * C_p_brass)
delta_T_brass_rad2_cold = delta_U_rad_2_cold /((1-fraction) * M_rod * C_p_brass)
print("--------")
print(f"50-300s interval (heat lost on total bar): {delta_T_brass_rad2}")
print(f"50-300s interval (heat lost on hot zone): {delta_T_brass_rad2_hot}")
print(f"50-300s interval (heat lost on cold zone): {delta_T_brass_rad2_cold}")

#300-1200s hot part at 308.15, cold part at 303.15K
P3_high = power_radiated_rod(L * fraction, d, epsilon = emissivity, T_sys = 308.15)
P3_low = power_radiated_rod(L * (1-fraction), d, epsilon = emissivity, T_sys = 303.15)
delta_U_rad_3_hot = 900 * P2_high
delta_U_rad_3_cold = 900 * P2_low
delta_U_rad_3 = delta_U_rad_3_hot + delta_U_rad_3_cold
delta_T_brass_rad3 = delta_U_rad_3 /(M_rod * C_p_brass)
delta_T_brass_rad3_hot = delta_U_rad_3_hot /(fraction * M_rod * C_p_brass)
delta_T_brass_rad3_cold = delta_U_rad_3_cold /((1-fraction) * M_rod * C_p_brass)
print("--------")
print(f"300-1200s interval (heat lost on total bar): {delta_T_brass_rad3}")
print(f"300-1200s interval (heat lost on hot zone): {delta_T_brass_rad3_hot}")
print(f"300-1200s interval (heat lost on cold zone): {delta_T_brass_rad3_cold}")


# %% [markdown]
# all are overestimates. Later times are gross overestimates. Real losses will be much smaller as sup has been used to bound the values. Real temperatures will decay much faster and use a lower temperature for radiation. Finally convection estimate based on air volume being replaced fully twice and heated during the experiment. Again gross overestimate.

# %%
print(f"change in rod temperature through convection with air: {2 * delta_T_brass}")


# %%
def celsius_to_kelvin(C):
    return 273.15 + C


# %%
#prediction for the first measurement
T_room_1 = 22.5 #in celcsius
T_room_1 = celsius_to_kelvin(T_room_1)
T_source_1 = 80
T_source_1 = celsius_to_kelvin(T_source_1)
N = 6
L = 0.3
d = 0.025 #diameter of rod
alpha = 0.34e-4 #conductivity of brass
t_final = 300

temp = T_room_1 * np.ones(N)
T_initial_1 = temp
T_initial_1[0] = T_source_1
print(f"initial temperature distribution = {T_initial_1}")
delta_x = L / N
x = delta_x * np.arange(N) + (delta_x/2) #get a vector with the positions implemented
coeffs = get_coeffs_cos_basis(T_initial_1, x)
coeffs_50 = lossless_propagation_1D(coeffs, t = 50) #propagate coeffs in time
coeffs_300 = lossless_propagation_1D(coeffs, t = 300) #propagate coeffs in time
coeffs_1200 = lossless_propagation_1D(coeffs, t = 1200) #propagate coeffs in time
print(f"intial cos basis coefficients: {np.round(coeffs, 2)}")
print(f"t = 50 cos basis coefficients: {np.round(coeffs_50, 3)}") 
print(f"t = 300 cos basis coefficients: {np.round(coeffs_300, 3)}")
print(f"t = 1200 cos basis coefficients: {np.round(coeffs_1200, 3)}")

T_min = T_room_1; T_max = T_source_1
fig, ax = plt.subplots(2, 2, figsize = (10, 10))
ax[0,0].scatter(x, T_initial)
ax[0,0].set(ylim = (T_min -5, T_max + 5))
ax[0,0].set_xlabel("position on rod [m]")
ax[0,0].set_ylabel("Temperature at sensor [K]")
ax[0,0].set_title(f"Temperature distribution at t = {0}")

ax[0,1].scatter(x, evaluate_cosine_sum(coeffs_50, x, L = L))
ax[0,1].set(ylim = (T_min -5, T_max + 5))
ax[0,1].set_xlabel("position on rod [m]")
ax[0,1].set_ylabel("Temperature at sensor [K]")
ax[0,1].set_title(f"Temperature distribution at t = {50}")

ax[1,0].scatter(x, evaluate_cosine_sum(coeffs_300, x, L = L))
ax[1,0].set(ylim = (T_min -5, T_max + 5))
ax[1,0].set_xlabel("position on rod [m]")
ax[1,0].set_ylabel("Temperature at sensor [K]")
ax[1,0].set_title(f"Temperature distribution at t = {300}")

ax[1,1].scatter(x, evaluate_cosine_sum(coeffs_1200, x, L = L))
ax[1,1].set(ylim = (T_min -5, T_max + 5))
ax[1,1].set_xlabel("position on rod [m]")
ax[1,1].set_ylabel("Temperature at sensor [K]")
ax[1,1].set_title(f"Temperature distribution at t = {1200}")
plt.show()

# %%
measurement_1 = np.loadtxt("Series_002.csv", skiprows = 1, delimiter = ',', usecols=range(1, 8))
#print(measurement_1)
t_1 = measurement_1[:, 0]
T_1 = measurement_1[:, 1:7]
T_1 = celsius_to_kelvin(T_1)

T_initial_1 = T_1[0] #for measurement
t_eval = [t_1[0], t_1[9], t_1[53], t_1[-1]]
coeffs = get_coeffs_cos_basis(T_initial_1, x)
coeffs_0 = lossless_propagation_1D(coeffs, t = t_eval[0]) #propagate coeffs in time
coeffs_50 = lossless_propagation_1D(coeffs, t = t_eval[1]) #propagate coeffs in time
coeffs_300 = lossless_propagation_1D(coeffs, t = t_eval[2]) #propagate coeffs in time
coeffs_1200 = lossless_propagation_1D(coeffs, t = t_eval[3]) #propagate coeffs in time

T_min = T_1[0, 5]
T_max = T_1[0, 0]
fig, ax = plt.subplots(2, 2, figsize = (10, 10))


ax[0,0].scatter(x, T_1[0])
ax[0,0].plot(x, evaluate_cosine_sum(coeffs_0, x, L = L), linestyle = '--', color = 'k')
ax[0,0].set(ylim = (T_min -5, T_max + 5))
ax[0,0].set_title(f"Temperature distribution at t = {t_eval[0]}")


ax[0,1].scatter(x, T_1[9])
ax[0,1].plot(x, evaluate_cosine_sum(coeffs_50, x, L = L), linestyle = '--', color = 'k')
ax[0,1].set(ylim = (T_min -5, T_max + 5))
ax[0,1].set_title(f"Temperature distribution at t = {t_eval[1]}")

ax[1,0].scatter(x, T_1[53])
ax[1,0].set(ylim = (T_min -5, T_max + 5))
ax[1,0].plot(x, evaluate_cosine_sum(coeffs_300, x, L = L), linestyle = '--', color = 'k')
ax[1,0].set_title(f"Temperature distribution at t = {t_eval[2]}")

ax[1,1].scatter(x, T_1[-1])
ax[1,1].set(ylim = (T_min -5, T_max + 5))
ax[1,1].plot(x, evaluate_cosine_sum(coeffs_1200, x, L = L), linestyle = '--', color = 'k')
ax[1,1].set_title(f"Temperature distribution at t = {t_eval[3]}")

#estimation of heat loss
T_meas = np.array([T_1[9], T_1[53], T_1[-1]])
coeff_times = [coeffs_50, coeffs_300, coeffs_1200]
T_pred = np.array([evaluate_cosine_sum(coeff, x, L = L) for coeff in coeff_times])
diff_T = T_pred - T_meas
delta_U = [np.sum(diff_T[i] * (M_rod * C_p_brass)) for i in range(3)]
print(f"energy lost = {delta_U} J")
print(f"Measured temperatures = {T_meas}")
print(f"predicted temperatures = {T_pred}")

# %%
#estimation of density of the rods (no uncertainties)
#all units in cm and g in this subsection
d = 2.5
l_long = 25.1
width_ridge = 0.1
height_ridge = 0.06
radius_hole = 0.1
depth_hole = 1.2
V_long_naive = pi * (d/2)**2 * l_long - pi * (d/2 - width_ridge)**2 * height_ridge #volume of rod with correction for indent
hole_correction_long = 5 * pi * radius_hole**2 * depth_hole
V_long = V_long_naive - hole_correction_long
print(f"Volume of long rod = {V_long}")
weight_long = 1035 #in grams
density_long = weight_long / V_long #g/cm^3
print(f"density of long rod = {density_long}") #actual around 8.47

l_short = 5.0
height_extrusion = 0.1
d_extrusion = 2.3
V_short_naive = pi * (d/2)**2 * (l_short - height_extrusion) + height_extrusion * pi * (d_extrusion/2)**2
hole_correction_short = depth_hole * pi * radius_hole**2
V_short = V_short_naive - hole_correction_short
print(f"Volume of short rod = {V_short}")
weight_short = 206 #in grams
density_short = weight_short / V_short #g/cm^3
print(f"density of long rod = {density_short}") #actual around 8.47
