# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -jupyter
#     formats: ipynb,py:hydrogen
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import scipy

# %%
"""
SHOPPING LIST:
- Function that integrates cos^n over an area of the unit sphere (pyramidal, see sketch)
- Function that returns the expected count rate
- I/O reader, then scipy.curve_fit for n in the above:
    x Read the CSV files
    x Convert angles to radian
    x Determine the opening angle based on inter-detector distance
    x 
"""

"""
SPECIFICATIONS:
Our length units MUST be cm. Every variable must give units as such.
Our time units may vary and MUST be specified in the docstring. This is to prevent floating-point annihilation issues.
TODO: Specify time units in all functions
"""

# %%
SIDE_LENGTH = 5
PLANE_DIST = 10


def area_integral(n: float = 2.0, theta: float = 0, distance: float = PLANE_DIST):
    """
    For a cos^{n} distribution over the unit sphere, return the expected count rate of two planar detectors in parallel with distance {distance} between them.

    The count rate for two points in a line: Consider an area dA1 on the lower plane at x1, and an area dA2 on the upper plane at x2. Then a line from the atmosphere through both areas sees the lower area as d/r dA1; dA2 as seen from x1 subtends the solid angle (d/r) dA2 / r^2. Thus, we need to scale cos^n by d**2/r**4.

    The area arguments only depend on the relative offsets, so we can simply double-integrate over the x- and y-offsets.
    TODO change to Gauss-Legendre quadrature. https://stackoverflow.com/questions/77002315/get-2d-gauss-legendre-quadrature-points-and-weights-using-numpy .
    """
    a = SIDE_LENGTH
    d = distance

    def integrand(dy, dx):
        return (
            (a - np.abs(dx))
            * (a - np.abs(dy))
            * (d / (dx**2 + dy**2 + d**2)) ** 2
            * np.cos(theta) ** n
        )

    return scipy.integrate.dblquad(integrand, -a, a, -a, a)[0]


# %%
WINDOW = 30e-3  # 30 us = 0.03 ms = 30e-6 s


def accidental_rate(singles_rate=0.5, window=WINDOW):
    """
    Returns the rate of accidentals from both detectors as given in the source.
    singles_rate must have the same unit as WINDOW.
    Returns units of WINDOW^{-1}
    """
    return 2 * singles_rate**2 * window


# %%
ACCIDENTALS = 0.1
INTENSITY = 170  # Per HOUR


def count_rate(
    n: float = 2,
    intensity: float = 170,  # In units of HOUR^-1
    accidentals: float = ACCIDENTALS,  # In units of HOUR^-1
    timespan: float = 1,  # In units of HOUR
    theta: float = 0,
    distance: float = 10,
):
    """
    Returns the expected count rate for an exponent {n}, maximal {intensity} of only muons and {accidental} rate over a given {timespan} in HOURS, when measuring over {theta} with {distance} between planes.
    """
    return timespan * (intensity * area_integral(n, theta, distance) + accidentals)


# %%
def read_one(filename: str):
    """
    Reads a single filename.
    """
    data = pd.read_csv(
        filename,
        delimiter=" ",
        header=5,
        index_col=0,
        names=["time", "adc", "sipm", "deadtime", "temp"],
    )  # Time in ms
    return data


def read_all(folder: str):
    """
    TODO
    Reads all files. Obtains distance, etc.
    """


# %%
sample = read_one("data/examples/Run2_S.TXT")
counts = len(sample.index)
# print(counts / sample.time.iloc[-1], accidental_rate(0.05))
SINGLES_RATE = scipy.optimize.least_squares(
    lambda x: accidental_rate(x) - counts / sample.time.iloc[-1], 0.05
).x
ACCIDENTALS = accidental_rate(SINGLES_RATE) * 1000 * 3600
print(ACCIDENTALS)


# %%
RUN_ANGLES_DEG = [0, 90, 15, 60, 30, 7.5, 45, 75, 45, 45, 45]

for i, theta in enumerate(RUN_ANGLES_DEG):
    if i == 6:
        continue
    data = read_one(f"data/runs/Run{i + 1}_S.TXT")
    counts = len(data.index)
    duration = read_one(f"data/runs/Run{i + 1}_M.TXT")["time"].iloc[-1]
    print(counts / duration)
# %%
