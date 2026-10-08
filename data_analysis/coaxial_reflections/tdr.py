"""TDR analysis: find echoes in the step response and compute cable length, Gamma_L and Z_L."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as unc
import warnings
import os
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter

script_dir = os.getcwd()
data_dir = os.path.join(script_dir, "measured_data")


# ========== Acquired data and variables ==========

# Properties (length [m] and impedance [Ohm]) of coaxial cables
length_c1 = unc.ufloat(28.0, 0.05)
length_c2 = unc.ufloat(54.5, 0.1)
length_c3 = unc.ufloat(80.6, 0.04)

impedance_c12 = unc.ufloat(75, 3)
impedance_c3 = unc.ufloat(50, 2)

# speed of light [m/s]
c = 299792458


# Data ./measured_data/dielectric
dielectric_dir = os.path.join(data_dir, "dielectric")
folder_c1_open = os.path.join(dielectric_dir, "cable1_open")
folder_c1_short = os.path.join(dielectric_dir, "cable1_short")
folder_c2_open = os.path.join(dielectric_dir, "cable2_open")
folder_c2_short = os.path.join(dielectric_dir, "cable2_short")
folder_c3_open = os.path.join(dielectric_dir, "cable3_open")
folder_c3_short = os.path.join(dielectric_dir, "cable3_short")
folder_c3_matched = os.path.join(dielectric_dir, "cable3_matched")
folder_c12_open = os.path.join(dielectric_dir, "cable1-2_open")
folder_c12_short = os.path.join(dielectric_dir, "cable1-2_short")


# ========== Loading and preparing the data ==========


def load_data(path):
    """Read one scope CSV, returns (t, V)."""
    df = pd.read_csv(path, skiprows=2, names=["t", "V"])
    return df["t"].to_numpy(), df["V"].to_numpy()


def load_folder(folder):
    """Load all CSVs in a folder."""
    files = sorted(Path(folder).glob("*.csv"))
    t = load_data(files[0])[0]
    U_all = np.array([load_data(f)[1] for f in files])
    return t, U_all


def data_preparation(t, U_all, t_min=-0.5, t_max=5, t_base=-0.05):
    """Average, smooth, crop to [t_min, t_max] and subtract the baseline."""
    U_mean = U_all.mean(axis=0)
    U_smooth = savgol_filter(U_mean, window_length=11, polyorder=2)
    mask = (t > t_min) & (t < t_max)
    U = U_smooth[mask]
    t = t[mask]
    U = U - U[t < t_base].mean()  # baseline = signal before the step
    return t, U



# ========== Finding voltage jumps and plateaus ==========


def find_jumps(t, U, thr_per=0.05, min_sep=0.1):
    """Find steps in U as peaks in |dU/dt|.
    thr_per: percentage of max slope for find peaks."""
    dt = t[1] - t[0]
    dU = savgol_filter(U, window_length=11, polyorder=2, deriv=1, delta=dt)
    dU_abs = np.abs(dU)  # abs: catch positive and negative reflections
    thr = thr_per * np.max(dU_abs)  # ignore slopes below thr_per% (defeault=5%) of max
    idx, _ = find_peaks(dU_abs, height=thr, distance=max(1, int(min_sep / dt)))
    if len(idx) == 0:
        raise Exception("No jumps found.")
    print(f"Found {len(idx)} jumps.")
    return idx, dU, thr


def plateau_levels(t, U, idx, margin=0.02):
    """Mean voltage between consecutive jumps; margin excludes the edges."""
    edges = np.append(t[idx], t[-1])
    levels = []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (t > a + margin) & (t < b - margin)
        levels.append(U[mask].mean())
    return np.array(levels)



# ========== Calculations ==========


def find_ZL(t, U, idx, Z_0=50.0, Z_S=50.0, A=1):
    """Echo times, Gamma_L and Z_L from the first echo. A = attenuation."""
    t_p = t[idx]
    dt_echo = t_p[1:] - t_p[0]  # echos relative to the incident step
    levels = plateau_levels(t, U, idx)

    if len(levels) < 2:
        raise Exception("No Echo found.")

    print(f"Calculating Gamma_L and Z_L with Z_0 = {Z_0} and Z_S = {Z_S}.\n")
    Gamma_S = (Z_S - Z_0) / (Z_S + Z_0)

    V1 = levels[0]
    dV = np.diff(levels)
    Gamma_L = dV[0] / ((1 + Gamma_S) * A * V1)
    Z_L = Z_0 * (1 + Gamma_L) / (1 - Gamma_L)
    return dt_echo, Gamma_L, Z_L


def cal_velocity_and_dielectric(length_c, dt_open, dt_short):
    """Calculates group velocity [m/s] of the signal and dielectric constant of the cable."""
    rel_diff = abs(dt_open - dt_short) / dt_open
    if rel_diff > 0.01:
        warnings.warn(
            f"open/short differ by {rel_diff * 100:.1f} %", UserWarning
        )  # Warning if time difference of short and open is too big

    dt_mean = np.mean(dt_short, dt_open)

    v = 2.0 * length_c / dt_mean

    mu_r = 1.0  # assuming the insulator isn't magnetic

    dielectric = (1 / mu_r) * (c / v) ** 2

    return v, dielectric



# ========== Plotting ==========


def validation_plot(
    folder_path, plot_title, Z_0=50.0, Z_S=50.0, A=1, thr_per=0.05, min_sep=0.1
):
    """Run the full analysis on one folder and plot derivative + detected plateaus.
    thr_per: percentage of max slope for find peaks."""
    print("\n", 5 * "=", plot_title, 5 * "=")
    t, U_all = load_folder(folder_path)
    t, U = data_preparation(t, U_all)

    idx, dU, thr = find_jumps(t, U, thr_per, min_sep)

    fig, ax = plt.subplots(nrows=2, figsize=(7, 8), dpi=300)

    # Top: derivative with detection threshold
    ax[0].plot(t, dU, label="Derivative")
    ax[0].axhline(thr, color="green", ls="--", label="Threshold")
    ax[0].axhline(-thr, color="green", ls="--")
    ax[0].set_xlabel(r"t [$\mu$s]")
    ax[0].set_ylabel("dV/dt")
    ax[0].set_title(f"Derivative of Signal ({plot_title})")
    ax[0].grid()
    ax[0].legend()

    # Bottom: signal with detected edges and plateaus
    ax[1].plot(t, U, label="Data")

    for i in idx:
        lbl = "Detected edges" if i == idx[0] else None
        ax[1].axvline(t[i], color="orange", ls="--", label=lbl)

    levels = plateau_levels(t, U, idx)
    edges = np.append(t[idx], t[-1])
    for a, b, L in zip(edges[:-1], edges[1:], levels):
        lbl = "Plateaus" if L == levels[0] else None
        ax[1].hlines(L, a, b, color="red", ls="--", label=lbl)

    ax[1].set_xlabel(r"t [$\mu$s]")
    ax[1].set_ylabel("V")
    ax[1].set_title(f"Signal ({plot_title})")
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    try:
        dt_echo, Gamma_L, Z_L = find_ZL(t, U, idx, Z_0=Z_0, Z_S=Z_S, A=A)
        print("Time of echo(s):", dt_echo)
        print("Gamma_L = ", Gamma_L)
        print("Z_L = ", Z_L)

        v = 0.66 * 299.792458  # velocity factor 0.66 * c in m/us
        print("Cable length = ", v * dt_echo[0] / 2)
    except Exception as e:
        print("Analysis failed:", e)
        dt_echo = Gamma_L = Z_L = None

    return {
        "t": t,
        "U": U,
        "idx": idx,
        "levels": levels,
        "dt_echo": dt_echo,
        "Gamma_L": Gamma_L,
        "Z_L": Z_L,
    }


# ========== ... ==========


if __name__ == "__main__":
    com = input("""
    What do you want to do?
    [a] Validation plots export
    [b] 

    Choose:""")

    print(f"\nChoice: {com}")

    match com:
        case "a":
            
