"""
Diffusion Weighted Signal
Objective: Calculate the diffusion weighted signal in a voxel

Assumptions:
During the diffusion weighting time, effectively do not exchange between
spatially distinct fiber bundles within the voxel.

The radius-of-curvature of a fiber bundle is greater than the average diffusion displacement,
so there is no exchange between orientationally distinct sections of the same voxel fiber bundle.

The diffusion characteristics of fiber populations in a voxel can
represented by a positive, definite, symmetric rand-2 tensor.

Approach:
Simulate the orientation-dependent, diffusion-weighted signal in
a voxel as a function of the diffusion weighted factor, b-value and
the gradient orientation (also may include T1 and T2 relaxation) for
for a voxel with an isotropic compartment and two orthogonal fibers

Created	20220624	by T. H. Mareci
Edited  20220627    Added video of b-value dependence
Adapted 20220718    Python adaptation by Warren Boschen
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import time
from general_functions import *

start = time.time()

DT0 = dt(2.0e-3, 2.0e-3, 2.0e-3) # Rank-2 isotropic diagonal tensor, DT0 in units of mm^2/s (free water contribution)
DT =  dt(0.2e-3, 0.2e-3, 1.7e-3) # Single fiber rank-2 anisotropic diagonal tensor, DT1 in units of mm^2/s (oriented along the z-axis)
DT1 = rotate(DT, np.pi/2, np.pi/4, np.pi/4)  # Rotated DT. Angles correspond to rotation along the z-, y-, and z-axes respectively

"""
Video Initialization

Video 1: Surface plot of Sx_Sb_D0, Sy_Sb_D0, and Sz_Sb_D0
Video 2: Surface plot of Sx_Sb_D1, Sy_Sb_D1, and Sz_Sb_D1
Video 3: Surface plot of Sx_S_Sum, Sy_S_sum, and Sz_S_sum (weighted sum)
"""

# ani('D0', 1.0, DT0, 100)
ani('D1', 1.0, DT1, 100)
# ani_multi(1.0, [DT0, DT1], [0.5, 0.5], 100) #! The data blows up almost instantly.

end = time.time()
print(end - start)
# Time elapsed using PNGs as frames: ~14min
# Time elapsed using writer.grab_frame(): ~7min