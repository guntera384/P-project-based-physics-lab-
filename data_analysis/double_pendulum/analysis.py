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
#     display_name: Python (pendulum-env)
#     language: python
#     name: pendulum-env
# ---

# %%
import math

import numpy as np

# %%
from physics_tools.coordinates import double_pendulum_angles, double_pendulum_xy

dpxy = double_pendulum_xy
dpc = double_pendulum_angles

# %%
print(dpxy(0, 0))
print(dpxy(0, np.pi))
print(dpc(5, 0, right=True))
dpxy(*dpc(-10, 0, True))
