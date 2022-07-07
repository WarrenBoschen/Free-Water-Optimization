"""
Tensor Rotation
Objective: Generate a fiber population from one tensor using matrix rotations

Created 20220707 by Warren Boschen
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import ffmpeg

# Define a diagonal diffusion tensor based on its eigenvalues
def dt(xx, yy, zz):
    return np.diag([xx, yy, zz])

# Rotate a diagonal diffusion tensor to be aligned with
# the polar angle Theta and the azimuthal angle Phi
def rotate(tensor, theta, phi):
    return "WIP"