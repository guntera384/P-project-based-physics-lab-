import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as unc
import warnings
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "measured_data"
PLOT_DIR = SCRIPT_DIR / "validation_plots"


# ========== Acquired data and variables ==========

# Properties (length [m] and impedance [Ohm]) of coaxial cables
length_c1 = unc.ufloat(28.0, 0.05)
length_c2 = unc.ufloat(54.5, 0.1)
length_c3 = unc.ufloat(80.6, 0.04)
CABLE_LENGTHS = {
    "cable1": length_c1,
    "cable2": length_c2,
    "cable3": length_c3,
    "cable1-2": length_c1 + length_c2,
}

impedance_c12 = unc.ufloat(75, 3)
impedance_c3 = unc.ufloat(50, 2)

# speed of light [m/s]
c = 299792458

# Parameters
DEFAULTS = dict(Z_0=75.0, Z_S=50.0, A=1, thr_per=0.05, min_sep=0.1)
OVERRIDES = {
    "detective2": dict(thr_per=0.04),
    "cable3_matched": dict(Z_0=50.0),
    "cable3_open": dict(Z_0=50.0),
    "cable3_short": dict(Z_0=50.0, thr_per=0.07),
    "fork_2cables_both_short": dict(thr_per=0.07),
    "fork_2cables_cable1open_cable2matched": dict(thr_per=0.03, min_sep=0.2),
    "fork_2cables_cable1short_cable2matched": dict(thr_per=0.03, min_sep=0.2),
    "fork_2cables_cable1short_cable2open": dict(thr_per=0.07),
}


# ========== Loading and preparing the data ==========


def get_measurements():
    folders = sorted({p.parent for p in DATA_DIR.rglob("*.csv")})
    return [
        {
            "name": f.name,
            "group": f.parent.name,
            "path": f,
            **DEFAULTS,
            **OVERRIDES.get(f.name, {}),
        }
        for f in folders
    ]


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
    """Average, smooth, crop to [t_min, t_max] and subtract the baseline.
    Returns (t, U, U_raw); U_raw is the averaged but unsmoothed signal."""
    U_mean = U_all.mean(axis=0)
    dt = t[1] - t[0]
    U_smooth = savgol_filter(U_mean, window_length=int(0.02 / dt) | 1, polyorder=2)
    mask = (t > t_min) & (t < t_max)
    U = U_smooth[mask]
    U_raw = U_mean[mask]
    t = t[mask]
    U = U - U[t < t_base].mean()  # baseline = signal before the step
    U_raw = U_raw - U_raw[t < t_base].mean()
    return t, U, U_raw


# ========== Finding voltage jumps and plateaus ==========


def find_jumps(t, U, thr_per=0.05, min_sep=0.1):
    """Find steps in U as peaks in |dU/dt|.
    thr_per: percentage of max slope for find peaks."""
    dt = t[1] - t[0]
    dU = savgol_filter(
        U, window_length=int(0.02 / dt) | 1, polyorder=2, deriv=1, delta=dt
    )
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
    if rel_diff > 0.05:
        warnings.warn(
            f"open/short differ by {rel_diff * 100:.1f} %", UserWarning
        )  # Warning if time difference of short and open is too big

    dt_mean = (dt_short + dt_open) / 2

    v = 2.0 * length_c / dt_mean

    mu_r = 1.0  # assuming the insulator isn't magnetic

    dielectric = (1 / mu_r) * (c / v) ** 2

    return v, dielectric


# ========== Plotting ==========


def validation_plot(res, save_dir=PLOT_DIR):
    """Plot derivative + detected plateaus and save it in PLOT_DIR."""

    t, U, dU, idx, thr = res["t"], res["U"], res["dU"], res["idx"], res["thr"]
    plot_title = f"{res['group']}: {res['name']}"

    fig, ax = plt.subplots(nrows=2, figsize=(7, 8), dpi=300)

    # Top: derivative with detection threshold
    ax[0].plot(t, res["dU_raw"], color="lightgray", lw=0.5, label="Raw derivative")
    ax[0].plot(t, dU, label="Derivative")
    pad = 0.1 * (dU.max() - dU.min())
    ax[0].set_ylim(dU.min() - pad, dU.max() + pad)
    ax[0].axhline(thr, color="green", ls="--", label="Threshold")
    ax[0].axhline(-thr, color="green", ls="--")
    ax[0].set_xlabel(r"t [$\mu$s]")
    ax[0].set_ylabel("dV/dt")
    ax[0].set_title(f"Derivative of Signal ({plot_title})")
    ax[0].grid()
    ax[0].legend()

    # Bottom: signal with detected edges and plateaus
    ax[1].plot(t, res["U_raw"], color="lightgray", lw=0.5, label="Raw data")
    ax[1].plot(t, U, label="Smoothed data")

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

    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir / f"{res['group']}_{res['name']}.png")
    plt.close(fig)


# ========== Analysis ==========


def general_analysis(m):
    """Full analysis for one measurement dict. Returns a results dict."""
    t, U_all = load_folder(m["path"])
    t, U, U_raw = data_preparation(t, U_all)
    idx, dU, thr = find_jumps(t, U, m["thr_per"], m["min_sep"])
    levels = plateau_levels(t, U, idx)
    res = dict(
        name=m["name"],
        group=m["group"],
        t=t,
        U=U,
        U_raw=U_raw,
        dU=dU,
        dU_raw=np.gradient(U_raw, t),
        thr=thr,
        idx=idx,
        levels=levels,
        dt_echo=None,
        Gamma_L=None,
        Z_L=None,
    )
    try:
        res["dt_echo"], res["Gamma_L"], res["Z_L"] = find_ZL(
            t, U, idx, Z_0=m["Z_0"], Z_S=m["Z_S"], A=m["A"]
        )
    except Exception as e:
        print(f"  [{m['name']}] analysis failed: {e}")
    return res


# ========== Tasks (with help of AI/Claude) ==========


def task_validation_plots(measurements):
    for m in measurements:
        print(f"-> {m['name']}")
        try:
            validation_plot(general_analysis(m))
        except Exception as e:
            print(f"  [{m['name']}] skipped: {e}")
    print(f"Plots saved to {PLOT_DIR}")


def task_results_table(measurements):
    rows = []
    for m in measurements:
        try:
            r = general_analysis(m)
        except Exception as e:
            print(f"  [{m['name']}] skipped: {e}")
            continue
        rows.append(
            dict(
                name=r["name"],
                group=r["group"],
                Gamma_L=r["Gamma_L"],
                Z_L=r["Z_L"],
                dt_echo=None if r["dt_echo"] is None else r["dt_echo"][0],
            )
        )
    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv(Path(__file__).parent / "results.csv", index=False)


def task_dielectric(measurements):
    # first echo time [us] for every measurement that was selected
    dt = {}
    for m in measurements:
        try:
            r = general_analysis(m)
        except Exception as e:
            print(f"  [{m['name']}] skipped: {e}")
            continue
        if r["dt_echo"] is not None:
            dt[m["name"]] = r["dt_echo"][0]

    for cable, length in CABLE_LENGTHS.items():
        if f"{cable}_open" not in dt or f"{cable}_short" not in dt:
            print(f"  {cable}: open or short missing, skipped")
            continue
        v, eps = cal_velocity_and_dielectric(
            length,
            dt[f"{cable}_open"] * 1e-6,
            dt[f"{cable}_short"] * 1e-6,  # us -> s
        )
        print(f"  {cable}: v = {v:.4g} m/s = {v / c:.3f} c,  eps_r = {eps:.3f}")


TASKS = {
    "a": ("Export validation plots", task_validation_plots),
    "b": ("Results table (Gamma_L, Z_L) -> results.csv", task_results_table),
    "c": ("Dielectric constant per cable", task_dielectric),
}


def choose_measurements(all_m):
    for i, m in enumerate(all_m, 1):
        print(f"  {i:2d}  {m['group']:12s} {m['name']}")
    sel = input("Datasets [Enter = all / group / 1,4,7]: ").strip()
    if not sel:
        return all_m
    if sel in {m["group"] for m in all_m}:
        return [m for m in all_m if m["group"] == sel]
    return [all_m[int(i) - 1] for i in sel.split(",")]
