"""Fit the parameters of a coax cable to a measured pulse response.

The cable is modelled as an ngspice LTRA line in coax.cir. The netlist only
holds default .param values; run_ngspice() overrides them with `alterparam`,
runs a batch simulation and hands the result back as a numpy array.
fit_cable() then adjusts the cable parameters until the simulated pulse
matches the scope trace.

(Parts of this script were generated, and the entirety of this script was refactored, by Claude.)
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import astuple, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
NETLIST = HERE / "coax.cir"
DATA = HERE / "test_data"
DATA50 = DATA / "20260928-0002_50Ohm"
DATA0 = DATA / "20260928_Short"
DATAINF = DATA / "20260928-0002_Open_End"

PLOT = False  # Controls whether to plot the result of each run.

# The optimiser works in "human" units (ohm/m, µH/m, pF/m, m) so that all
# parameters are of similar magnitude; Cable stores SI units.
FIT_UNITS = np.array([1.0, 1e-6, 1e-12])
FIT_BOUNDS = ([0, 0, 0], [10, 10, 1000])

TFACTOR = 100
# Controls accuracy of ngspice simulation. Larger = more speed, less accuracy

# The global variable coax length is set in the ngspice file.


@dataclass(frozen=True)
class Cable:
    """LTRA line parameters in SI units, per metre where applicable."""

    res: float = 0.30965627  # ohm/m   (.param intres)
    ind: float = 0.44195154e-6  # H/m     (.param intind)
    cap: float = 67e-12  # F/m     (.param indcap)

    def netlist_params(self) -> dict[str, float]:
        return {
            "intres": self.res,
            "intind": self.ind,
            "indcap": self.cap,
        }

    @classmethod
    def from_vector(cls, x) -> Cable:
        return cls(*map(float, np.asarray(x, dtype=float) * FIT_UNITS))

    def to_vector(self) -> np.ndarray:
        return np.array(astuple(self)) / FIT_UNITS


@dataclass(frozen=True)
class Scope:
    """A measured trace: time in µs, voltage in V."""

    t: np.ndarray
    v: np.ndarray
    load: float = 50.0

    @classmethod
    def from_csv(cls, path: Path) -> Scope:
        t, v = np.loadtxt(path, delimiter=",", skiprows=2, usecols=(0, 1), unpack=True)
        return cls(t, v)

    @property
    def t0(self) -> float:
        return self.t[0]

    @property
    def span(self) -> float:
        return self.t[-1] - self.t[0]

    @property
    def dt(self) -> float:
        return float(np.diff(self.t).mean())


def run_ngspice(
    params: dict[str, float], commands: str, vectors: list[str]
) -> np.ndarray:
    """Run the netlist with `params` overridden and return the `wrdata` table.

    `commands` is the analysis line (e.g. "tran 1n 10u"); the columns of the
    result are those of ngspice's wrdata for `vectors`.
    """
    alter = "\n".join(f"alterparam {name} = {value}" for name, value in params.items())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = tmp / "out.txt"
        circuit = tmp / "run.cir"
        circuit.write_text(
            f"{NETLIST.read_text()}\n"
            ".control\n"
            "set wr_singlescale\n"
            f"{alter}\n"
            "reset\n"
            f"{commands}\n"
            f"wrdata {out} {' '.join(vectors)}\n"
            ".endc\n"
            ".end\n"
        )
        proc = subprocess.run(
            ["ngspice", "-b", str(circuit)], capture_output=True, text=True
        )
        # ngspice often exits with 0 even when the run failed, so check the output.
        if not out.exists():
            raise RuntimeError(
                f"ngspice produced no output.\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        return np.loadtxt(out)


def simulate_pulse(cable: Cable, scope: Scope, vmax: float = 1.0):
    """Simulate the pulse response; returns (time in µs on the scope axis, V(2))."""
    params = {
        **cable.netlist_params(),
        "zl": scope.load,
        "sinamp": 0,
        "pulseamp": vmax,
        "td": -scope.t0 * 1e-6,  # delay the edge so the simulation can start at t = 0
    }
    tstep, tstop = scope.dt * 1e-6 * TFACTOR, scope.span * 1e-6
    data = run_ngspice(params, f"tran {tstep} {tstop}", ["v(2)"])
    return data[:, 0] * 1e6 + scope.t0, data[:, 1]


def simulate_ac(cable: Cable, scope: Scope, vmax: float = 1.0, freq: float = 1e3):
    """AC sweep; returns (frequency, V(2)/V(23))."""
    params = {
        **cable.netlist_params(),
        "zl": scope.load,
        "sinamp": vmax,
        "pulseamp": 0,
        "freq": freq,
        "td": 0,
    }
    data = run_ngspice(params, "ac dec 10 1 300meg", ["v(2)", "v(23)"])
    f, v2, v23 = data[:, 0], data[:, 1] + 1j * data[:, 2], data[:, 3] + 1j * data[:, 4]
    return f, v2 / v23


def fit_cable(scope: Scope, start: Cable = Cable()):
    """Least-squares fit of the cable parameters to the measured pulse."""

    def residuals(x):
        ts, vs = simulate_pulse(Cable.from_vector(x), scope)
        sim = np.interp(scope.t, ts, vs)
        A = np.column_stack([sim, np.ones_like(sim)])
        coef, *_ = np.linalg.lstsq(A, scope.v, rcond=None)  # gain, offset
        return scope.v - A @ coef

    x0 = start.to_vector()
    return least_squares(
        residuals,
        x0,
        bounds=FIT_BOUNDS,
        x_scale=x0,
        diff_step=1e-3,  # ngspice output is only accurate to ~reltol (1e-3)
        verbose=0,
    )


def main(datafile) -> None:
    scope = Scope.from_csv(datafile)
    fit = fit_cable(scope)
    cable = Cable.from_vector(fit.x)
    # print(cable, f"cost={fit.cost:.3g}")
    if PLOT:
        t, v = simulate_pulse(cable, scope)
        sim = np.interp(scope.t, t, v)
        A = np.column_stack([sim, np.ones_like(sim)])
        coef, *_ = np.linalg.lstsq(A, scope.v, rcond=None)
        v = A @ coef

        _, ax = plt.subplots()
        ax.plot(scope.t, scope.v, label="measured")
        ax.plot(scope.t, v, label="fit")
        ax.set(xlabel="µs", ylabel="V(2)")
        ax.legend()
        plt.show()

    return cable.to_vector()


if __name__ == "__main__":
    results = []
    for file in DATA50.glob("**/*"):
        if file.is_file():
            # print(file)
            result = main(file)
            results.append(result)
    res50 = np.array(results).T
    print(np.mean(res50, axis=1), np.std(res50, axis=1))
    results = []
    for file in DATA0.glob("**/*"):
        if file.is_file():
            # print(file)
            result = main(file)
            results.append(result)
    res50 = np.array(results).T
    print(np.mean(res50, axis=1), np.std(res50, axis=1))
    results = []
    for file in DATAINF.glob("**/*"):
        if file.is_file():
            # print(file)
            result = main(file)
            results.append(result)
    res50 = np.array(results).T
    print(np.mean(res50, axis=1), np.std(res50, axis=1))
    real_cable = Cable.from_vector(np.mean(res, axis=1))
    f, ph = simulate_ac(real_cable, Scope.from_csv(DATA / "20260928-0002_50Ohm_01.csv"))
    ph = np.degrees(np.unwrap(np.angle(ph)))
    _, ax = plt.subplots()
    ax.semilogx(f / 1e6, ph)
    ax.set(xlabel="MHz", ylabel="phase of V(2) vs V(23) [deg]")
    plt.show()
