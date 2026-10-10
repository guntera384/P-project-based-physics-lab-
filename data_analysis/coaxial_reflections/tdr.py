import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as unc
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "measured_data"
PLOT_DIR_VALIDATION = SCRIPT_DIR / "validation_plots"
PLOT_DIR_FINAL = SCRIPT_DIR / "final_plots"

LATEX_TABULAR_TRUE = True

# Titles for final plot
LABELS = {
    # unknown terminations
    "detective1": "Circuit detective 1",
    "detective2": "Circuit detective 2",
    # single cables
    "cable1-2_open": "Cables 1 and 2 in series, open end",
    "cable1-2_short": "Cables 1 and 2 in series, short-circuited end",
    "cable1_open": "Cable 1, open end",
    "cable1_short": "Cable 1, short-circuited end",
    "cable2_open": "Cable 2, open end",
    "cable2_short": "Cable 2, short-circuited end",
    "cable3_matched": "Cable 3, matched termination (50 Ω)",
    "cable3_open": "Cable 3, open end",
    "cable3_short": "Cable 3, short-circuited end",
    # splitter (two cables in parallel)
    "fork_2cables_both_open": "Splitter: both cables open",
    "fork_2cables_both_short": "Splitter: both cables short-circuited",
    "fork_2cables_cable1open_cable2matched": "Splitter: cable 1 open, cable 2 matched",
    "fork_2cables_cable1open_cable2short": "Splitter: cable 1 open, cable 2 short-circuited",
    "fork_2cables_cable1short_cable2matched": "Splitter: cable 1 short-circuited, cable 2 matched",
    "fork_2cables_cable1short_cable2open": "Splitter: cable 1 short-circuited, cable 2 open",
}


# ========== Acquired data and variables ==========

# Properties (length [m] and impedance [Ohm]) of coaxial cables
length_c1 = unc.ufloat(28.0, 0.05)
length_c2 = unc.ufloat(54.5, 0.1)
length_c3 = unc.ufloat(80.6, 0.04)
length_shortener = unc.ufloat(0.22, 0.01)
CABLE_LENGTHS = {
    "cable1": length_c1,
    "cable2": length_c2,
    "cable3": length_c3,
    "cable1-2": length_c1 + length_c2,
}

impedance_c12 = unc.ufloat(75, 3)
impedance_c3 = unc.ufloat(50, 2)

# source impedance
Z_S = unc.ufloat(50, 5)
Z_S_fork = 1 / (1 / Z_S + 1 / impedance_c12)


# speed of light [m/s]
c = 299792458

# Parameters
DEFAULTS = dict(Z_0=impedance_c12, Z_S=Z_S, thr_per=0.05, min_sep=0.1)
OVERRIDES = {
    "detective2": dict(thr_per=0.04),
    "cable3_matched": dict(Z_0=impedance_c3),
    "cable3_open": dict(Z_0=impedance_c3),
    "cable3_short": dict(Z_0=impedance_c3, thr_per=0.07),
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
            **({"Z_S": Z_S_fork} if f.parent.name == "fork_2cables" else {}),
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


def plateau_levels(t, U, idx, margin=0.02, with_unc=False):
    """Mean voltage between consecutive jumps; margin excludes the edges.
    with_unc=True returns ufloats with the std of the plateau as uncertainty."""
    edges = np.append(t[idx], t[-1])
    levels = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = U[(t > a + margin) & (t < b - margin)]
        if with_unc:
            levels.append(unc.ufloat(seg.mean(), seg.std(ddof=1)))
        else:
            levels.append(seg.mean())
    return np.array(levels)


# ========== Calculations ==========


def find_ZL(t, U, idx, Z_0, Z_S):
    """Echo times, Gamma_L and Z_L from the first echo."""
    t_p = t[idx]
    dt_echo = t_p[1:] - t_p[0]  # echos relative to the incident step
    levels = plateau_levels(t, U, idx, with_unc=True)

    if len(levels) < 2:
        raise Exception("No Echo found.")

    print(f"Calculating Gamma_L and Z_L with Z_0 = {Z_0} and Z_S = {Z_S}.\n")
    Gamma_S = (Z_S - Z_0) / (Z_S + Z_0)

    V1 = levels[0]
    dV = np.diff(levels)
    Gamma_L = dV[0] / ((1 + Gamma_S) * V1)
    Z_L = Z_0 * (1 + Gamma_L) / (1 - Gamma_L)
    return dt_echo, Gamma_L, Z_L


def cal_velocity_and_dielectric(length_c, dt_open, dt_short, dt_sample):
    """Calculates group velocity [m/s] of the signal and dielectric constant of the cable."""
    # dt_sample is taken as the uncertainty
    v_open = 2 * length_c / unc.ufloat(dt_open, dt_sample)
    v_short = 2 * (length_c + length_shortener) / unc.ufloat(dt_short, dt_sample)
    v = (v_open + v_short) / 2
    sys = abs(v_open.n - v_short.n) / 2  # systematic uncertainty
    v = v + unc.ufloat(0, sys)
    eps = (c / v) ** 2

    return v, eps, v_open, v_short


# ========== Plotting ==========


def validation_plot(res, save_dir=PLOT_DIR_VALIDATION):
    """Plot derivative + detected plateaus and save it in PLOT_DIR_VALIDATION."""

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


def final_plot(res, save_dir=PLOT_DIR_FINAL):
    """Plot detected plateaus and save it in PLOT_DIR_FINAL."""

    t, U, dU, idx, thr = res["t"], res["U"], res["dU"], res["idx"], res["thr"]

    plot_title = f"{LABELS.get(res['name'], res['name'])}"

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    # signal with detected edges and plateaus
    ax.plot(t, U, label="Signal")

    levels = plateau_levels(t, U, idx)
    edges = np.append(t[idx], t[-1])

    for a, b, i in zip(np.concatenate(([0], levels[:-1])), levels[:], idx):
        lbl = "Detected steps" if i == idx[0] else None
        ax.vlines(t[i], a, b, color="orange", alpha=0.5, ls="-", label=lbl)

    for a, b, L in zip(edges[:-1], edges[1:], levels):
        lbl = "Plateau levels" if L == levels[0] else None
        ax.hlines(L, a, b, color="red", alpha=0.5, ls="-", label=lbl)

    ax.set_xlabel(r"Time $t$ [µs]")
    ax.set_ylabel(r"Voltage $V(0,t)$ [V]")
    ax.set_title(f"Measured signal ({plot_title})")
    ax.grid()
    U_mid = (U.max() - U.min()) / 2
    if levels[-1] > U_mid:
        chosen_loc = "lower right"
    else:
        chosen_loc = "upper right"
    plt.legend(loc=chosen_loc)
    plt.tight_layout()

    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir / f"{res['group']}_{res['name']}.pdf")
    plt.close(fig)


# ========== Analysis ==========


def general_analysis(m):
    """Full analysis for one measurement dict. Returns a results dict."""
    t, U_all = load_folder(m["path"])
    t, U, U_raw = data_preparation(t, U_all)
    dt_sample = (t[1] - t[0]) * 1e-6  # t in us -> s
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
        dt_sample=dt_sample,
        Gamma_L=None,
        Z_L=None,
    )
    try:
        res["dt_echo"], res["Gamma_L"], res["Z_L"] = find_ZL(
            t, U, idx, Z_0=m["Z_0"], Z_S=m["Z_S"]
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
    print(f"Plots saved to {PLOT_DIR_VALIDATION}")


def task_final_plots(measurements):
    for m in measurements:
        print(f"-> {m['name']}")
        try:
            final_plot(general_analysis(m))
        except Exception as e:
            print(f"  [{m['name']}] skipped: {e}")
    print(f"Plots saved to {PLOT_DIR_FINAL}")


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
    if LATEX_TABULAR_TRUE:  # LaTeX Tabular output toggle
        df_cleaned = df.replace("_", " ", regex=True)
        df_renamed = df_cleaned.rename(
            columns={
                "name": "Setup",
                "Gamma_L": "$\\Gamma_L$",
                "dt_echo": "$\\Delta t_{\\text{echo}}$ [$\\mu$s]",
            }
        )
        print(
            df_renamed.to_latex(
                columns=[
                    "Setup",
                    "$\\Gamma_L$",
                    "$\\Delta t_{\\text{echo}}$ [$\\mu$s]",
                ],
                index=False,
                float_format="{:0.2e}".format,
                escape=False,
            )
        )
    else:
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
        v, eps, v_open, v_short = cal_velocity_and_dielectric(
            length,
            dt[f"{cable}_open"] * 1e-6,
            dt[f"{cable}_short"] * 1e-6,  # us -> s
            r["dt_sample"],
        )
        print(
            f"  {cable}: v_open = {v_open:.4g} m/s = {v_open / c:.3f} c,\t v_short = {v_short:.4g} m/s = {v_short / c:.3f} c,\t v = {v:.4g} m/s = {v / c:.3f} c,\t  eps_r = {eps:.3f}"
        )


def task_fork_times(measurements):
    """Echo times of the fork measurements compared with the single cables."""
    rows = []
    all_m = {m["name"]: m for m in get_measurements()}

    # reference: single cables
    for name in ["cable1_open", "cable1_short", "cable2_open", "cable2_short"]:
        dt = general_analysis(all_m[name])["dt_echo"]
        rows.append(dict(name=name, dt1=dt[0], dt2=None))

    # fork measurements
    for m in measurements:
        if m["group"] != "fork_2cables":
            continue
        r = general_analysis(m)
        if r["dt_echo"] is None or len(r["dt_echo"]) < 2:
            print(f"  [{m['name']}] fewer than 2 echoes found, skipped")
            continue
        rows.append(dict(name=m["name"], dt1=r["dt_echo"][0], dt2=r["dt_echo"][1]))

    df = pd.DataFrame(rows)
    if LATEX_TABULAR_TRUE:  # LaTeX Tabular output toggle
        df_cleaned = df.replace("_", " ", regex=True)
        df_renamed = df_cleaned.rename(
            columns={
                "name": "Setup",
                "dt1": "$\\Delta t_1$ [$\\mu$s]",
                "dt2": "$\\Delta t_2$ [$\\mu$s]",
            }
        )
        print(df_renamed.to_latex(index=False, float_format="%.3f", na_rep="--"))
    else:
        print(df.to_string(index=False))


def task_detective_times(measurements):
    """Echo times of the detective measurements."""
    rows = []
    for m in measurements:
        if m["group"] != "detective":
            continue
        r = general_analysis(m)
        if r["dt_echo"] is None:
            print(f"  [{m['name']}] no echo found, skipped")
            continue
        row = dict(name=m["name"])
        for k, dt in enumerate(r["dt_echo"], 1):
            row[f"dt{k}"] = dt
        rows.append(row)

    df = pd.DataFrame(rows)
    if LATEX_TABULAR_TRUE:  # LaTeX Tabular output toggle
        df_cleaned = df.replace("_", " ", regex=True)
        df_renamed = df_cleaned.rename(
            columns={
                "name": "Setup",
                **{
                    f"dt{k}": f"$\\Delta t_{k}$ [$\\mu$s]"
                    for k in range(1, df.shape[1])
                },
            }
        )
        print(df_renamed.to_latex(index=False, float_format="%.3f", na_rep="--"))
    else:
        print(df.to_string(index=False))


TASKS = {
    "b": ("Export validation plots", task_validation_plots),
    "c": ("Results table (Gamma_L, Z_L) -> results.csv", task_results_table),
    "d": ("Dielectric constant per cable", task_dielectric),
    "e": ("Fork: echo times compared with single cables", task_fork_times),
    "f": ("Detective: echo times", task_detective_times),
    "g": ("Export final plots", task_final_plots),
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
