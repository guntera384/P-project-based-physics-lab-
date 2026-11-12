# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -jupyter
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import math

import numpy as np

# %% [markdown]
#
# %%
from physics_tools.coordinates import double_pendulum_angles, double_pendulum_xy

dpxy = double_pendulum_xy
dpc = double_pendulum_angles

# %%
print(norm(array(dpxy(radians(80), radians(80)))))
print(dpxy(0, 0))
print(dpxy(0, np.pi))
print(dpc(5, 0, right=True))
dpxy(*dpc(-10, 0, True))
