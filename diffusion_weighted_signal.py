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
Edited  20220627  Added video of b-value dependence
Adapted 20220628    Python adaptation by Warren Boschen
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import cv2

# Compartment fractions/weightings
f_D0 = 1/4.0       # isotropic volume fraction
f_D1 = 1/4.0       # x-fiber volume fraction
f_D2 = 1/4.0       # y-fiber volume fraction
f_D3 = 1/4.0       # z-fiber volume fraction

# Initial signal without diffusion weighting (arbritary units)
S0 = 1.0

# Fiber 0 rank-2 isotropic diagonal tensor, DT0 in units of mm^2/s
# Free water contribution

D0_xx = 2.0e-3
D0_xy = 0.0
D0_xz = 0.0
D0_yy = 2.0e-3
D0_yz = 0.0
D0_zz = 2.0e-3
DT0 = np.array([D0_xx, D0_xy, D0_xz], [D0_xy, D0_yy, D0_yz],[D0_xz, D0_yz, D0_zz])

# Estimated maximum DW signal for D0 (used for scaling surface plot)
D0_avg = (D0_xx + D0_yy + D0_zz)/3
# D0_S_max = S0*np.exp(-b_value*D0_avg)

# Fiber 1 rank-2 isotropic diagonal tensor, DT1 in units of mm^2/s
# Oriented along the x-axis

D1_xx = 1.7e-3
D1_xy = 0.0
D1_xz = 0.0
D1_yy = 0.2e-3
D1_yz = 0.0
D1_zz = 0.2e-3
DT1 = np.array([D1_xx, D1_xy, D1_xz], [D1_xy, D1_yy, D1_yz],[D1_xz, D1_yz, D1_zz])

# Estimated maximum DW signal for D1 (used for scaling surface plot)
# Assuming D1_zz is the lowest rate of diffusion
# D1_S_max = S0*np.exp(-b_value*D1_zz)

# Fiber 2 rank-2 isotropic diagonal tensor, DT2 in units of mm^2/s
# Oriented along the y-axis

D2_xx = 0.2e-3
D2_xy = 0.0
D2_xz = 0.0
D2_yy = 1.7e-3
D2_yz = 0.0
D2_zz = 0.2e-3
DT2 = np.array([D2_xx, D2_xy, D2_xz], [D2_xy, D2_yy, D2_yz],[D2_xz, D2_yz, D2_zz])

# Estimated maximum DW signal for D2 (used for scaling surface plot)
# Assuming D2_zz is the lowest rate of diffusion
# D2_S_max = S0*np.exp(-b_value*D2_zz)

# Fiber 3 rank-2 isotropic diagonal tensor, DT3 in units of mm^2/s
# Oriented along the z-axis

D3_xx = 0.2e-3
D3_xy = 0.0
D3_xz = 0.0
D3_yy = 0.2e-3
D3_yz = 0.0
D3_zz = 1.7e-3
DT3 = np.array([D3_xx, D3_xy, D3_xz], [D3_xy, D3_yy, D3_yz],[D3_xz, D3_yz, D3_zz])

# Estimated maximum DW signal for D3 (used for scaling surface plot)
# Assuming D3_zz is the lowest rate of diffusion
# D3_S_max = S0*np.exp(-b_value*D3_zz)

# Gradient unit vector directions
Thetas = 100          # Number of polar angle iterations
dtheta = np.pi/Thetas # Polar angle iteration
Phis = 100            # Number of azimuthal angle iterations
dphi = (2*np.pi)/Phis # Azimuthal angle iteration

# Calculate diffusion weighted signal and surface plot values

# Preallocate/initialize ararys to zeros to accelerate calculation
Sb_D0 = np.zeros(Thetas, Phis)
Sb_D1 = np.zeros(Thetas, Phis)
Sb_D2 = np.zeros(Thetas, Phis)
Sb_D3 = np.zeros(Thetas, Phis)
Sx_Sb_D0 = np.zeros(Thetas, Phis)
Sy_Sb_D0 = np.zeros(Thetas, Phis)
Sz_Sb_D0 = np.zeros(Thetas, Phis)
Sx_Sb_D1 = np.zeros(Thetas, Phis)
Sy_Sb_D1 = np.zeros(Thetas, Phis)
Sz_Sb_D1 = np.zeros(Thetas, Phis)
Sx_Sb_D2 = np.zeros(Thetas, Phis)
Sy_Sb_D2 = np.zeros(Thetas, Phis)
Sz_Sb_D2 = np.zeros(Thetas, Phis)
Sx_Sb_D3 = np.zeros(Thetas, Phis)
Sy_Sb_D3 = np.zeros(Thetas, Phis)
Sz_Sb_D3 = np.zeros(Thetas, Phis)
Sx_S_sum = np.zeros(Thetas, Phis)
Sy_S_sum = np.zeros(Thetas, Phis)
Sz_S_sum = np.zeros(Thetas, Phis)
S_sum = np.zeros(Thetas, Phis)

# Establish video framework
frameSize = (500, 500)
out = cv2.VideoWriter('Sig_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

# Set video b-value parameters (for single b-value set N_bvalues to 1)
N_bvalues = 50      # number of b-values 
b_value_increm = 50 # b-value size increment in units of s/mm^2
b_value = 0.5e3     # initial b-value in units of s/mm^2

for k in range(N_bvalues):
    Theta = 0   # Intialize polar angle
    for i in range(Theta, Thetas+1, dtheta):
        Phi = 0 # Initialize azimuthal angle
        for j in range(Phi, Phis+1, dphi):
            # Create gradient unit vector (g_unit) and traponse (g_unit_T)
            g_unit = np.array([[np.cos(Phi)*np.sin(Theta)], 
                               [np.sin(Phi)*np.sin(Theta)], 
                               [np.cos(Theta)]])
            g_unit_T = np.transpose(g_unit)
            
            # DW signal for D0-3
            Sb_D0[i, j] = S0*np.exp(-b_value * g_unit_T * DT0 * g_unit)
            Sb_D1[i, j] = S0*np.exp(-b_value * g_unit_T * DT1 * g_unit)
            Sb_D2[i, j] = S0*np.exp(-b_value * g_unit_T * DT2 * g_unit)
            Sb_D3[i, j] = S0*np.exp(-b_value * g_unit_T * DT3 * g_unit)
            
            # Surface plot values for D0
            Sx_Sb_D0[i, j] = np.cos(Phi)*np.sin(Theta)*Sb_D0[i, j]
            Sy_Sb_D0[i, j] = np.sin(Phi)*np.sin(Theta)*Sb_D0[i, j]
            Sz_Sb_D0[i, j] = np.cos(Theta)*Sb_D0[i, j]
            
            # Surface plot values for D1
            Sx_Sb_D1[i, j] = np.cos(Phi)*np.sin(Theta)*Sb_D1[i, j]
            Sy_Sb_D1[i, j] = np.sin(Phi)*np.sin(Theta)*Sb_D1[i, j]
            Sz_Sb_D1[i, j] = np.cos(Theta)*Sb_D1[i, j]
            
            # Surface plot values for D2
            Sx_Sb_D2[i, j] = np.cos(Phi)*np.sin(Theta)*Sb_D2[i, j]
            Sy_Sb_D2[i, j] = np.sin(Phi)*np.sin(Theta)*Sb_D2[i, j]
            Sz_Sb_D2[i, j] = np.cos(Theta)*Sb_D2[i, j]
            
            # Surface plot values for D3
            Sx_Sb_D3[i, j] = np.cos(Phi)*np.sin(Theta)*Sb_D3[i, j]
            Sy_Sb_D3[i, j] = np.sin(Phi)*np.sin(Theta)*Sb_D3[i, j]
            Sz_Sb_D3[i, j] = np.cos(Theta)*Sb_D3[i, j]
            
            #DW signal for weighted sum of D0-3
            S_sum[i, j] = f_D0*Sb_D0[i, j] + f_D1*Sb_D1[i, j] + f_D2*Sb_D2[i, j] + f_D3*Sb_D3[i, j]
            
            # Surface plot values for weighted sum of D0-3
            Sx_S_sum[i, j] = np.cos(Phi)*np.sin(Theta)*S_sum[i, j]
            Sy_S_sum[i, j] = np.sin(Phi)*np.sin(Theta)*S_sum[i, j]
            Sz_S_sum[i, j] = np.cos(Theta)*S_sum[i, j]

# Surface plot of DW signal for weighted sum of D0-3

for a in range(0, 255):
    
    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
    X = np.arange(-0.7, 0.7, 0.07)
    Y = np.arange(-0.7, 0.7, 0.07)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-0.7, 0.7)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    fig.colorbar(surf, shrink=0.5, aspect=5)