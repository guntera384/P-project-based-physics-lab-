# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as unc

# %%
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "measured_data"
PLOT_DIR_FINAL = SCRIPT_DIR / "final_plots"

# %%
g = 1
m_1 = 1
m_2 = 1
L_1 = 1
L_2 = 1


# potential energy
def pot_energy(theta_1, theta_2):
    V_1 = -m_1 * g * L_1 * np.cos(theta_1)
    V_2 = -m_2 * g * (L_1 * np.cos(theta_1) + L_2 * np.cos(theta_2))
    return V_1 + V_2


# %%
def plot_(res, save_dir=PLOT_DIR_FINAL):

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

    ax.set_xlabel(r"$\\theta_1$")
    ax.set_ylabel(r"$\\theta_2$")
    ax.set_title("Initial position flip plot")
    ax.grid()
    plt.legend()
    plt.tight_layout()

    # save_dir.mkdir(exist_ok=True)
    # fig.savefig(save_dir / f"{res['group']}_{res['name']}.pdf")
    # plt.close(fig)
    plt.show()
