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
Adapted 20220711    Python adaptation by Warren Boschen
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from general_functions import *
import ffmpeg

DT0 = dt(2.0e-3, 2.0e-3, 2.0e-3) # Fiber 0 rank-2 isotropic diagonal tensor, DT0 in units of mm^2/s (free water contribution)
DT1 = dt(1.7e-3, 0.2e-3, 0.2e-3) # Fiber 1 rank-2 anisotropic diagonal tensor, DT1 in units of mm^2/s (oriented along the x-axis)
DT2 = dt(0.2e-3, 1.7e-3, 0.2e-3) # Fiber 2 rank-2 anisotropic diagonal tensor, DT2 in units of mm^2/s (oriented along the y-axis)
DT3 = dt(0.2e-3, 0.2e-3, 1.7e-3) # Fiber 3 rank-2 anisotropic diagonal tensor, DT3 in units of mm^2/s (oriented along the z-axis)

# Estimated maximum DW signal for D0 (used for scaling surface plot)
D0_avg = (DT0[0, 0] + DT0[1,1] + DT0[2, 2])/3
# D0_S_max = S0*np.exp(-b_value*D0_avg)

# Compartment fractions/weightings
f_D0 = 1/4.0       # isotropic volume fraction
f_D1 = 1/4.0       # x-fiber volume fraction
f_D2 = 1/4.0       # y-fiber volume fraction
f_D3 = 1/4.0       # z-fiber volume fraction

# Initial signal without diffusion weighting (arbritary units)
S0 = 1.0

# Gradient unit vector directions
Thetas = np.linspace(0, np.pi, 100)   # Polar angle iterations
Phis = np.linspace(0, 2*np.pi, 100)   # Azimuthal angle iterations

# Calculate diffusion weighted signal and surface plot values

# Preallocate/initialize ararys to zeros to accelerate calculation
Sb_D0 = np.zeros((100, 100))
Sb_D1 = np.zeros((100, 100))
Sb_D2 = np.zeros((100, 100))
Sb_D3 = np.zeros((100, 100))
Sx_Sb_D0 = np.zeros((100, 100))
Sy_Sb_D0 = np.zeros((100, 100))
Sz_Sb_D0 = np.zeros((100, 100))
Sx_Sb_D1 = np.zeros((100, 100))
Sy_Sb_D1 = np.zeros((100, 100))
Sz_Sb_D1 = np.zeros((100, 100))
Sx_Sb_D2 = np.zeros((100, 100))
Sy_Sb_D2 = np.zeros((100, 100))
Sz_Sb_D2 = np.zeros((100, 100))
Sx_Sb_D3 = np.zeros((100, 100))
Sy_Sb_D3 = np.zeros((100, 100))
Sz_Sb_D3 = np.zeros((100, 100))
Sx_S_sum = np.zeros((100, 100))
Sy_S_sum = np.zeros((100, 100))
Sz_S_sum = np.zeros((100, 100))
S_sum = np.zeros((100, 100))

# Set b-value parameters (for single b-value set N_bvalues to 1)
b_value = 500
b_value_stop = 3000
b_value_step = 50

counter_array = ['%.2d' % a for a in range(50)] # Generate array used for labelling video frames with the number of b-values
counter = 0
for k in range(b_value, b_value_stop, b_value_step): # Starting b-value, ending b-value, b-value increment size. All in s/mm^2
    for i in range(len(Thetas) - 1):
        for j in range(len(Phis) - 1):
            # Create gradient unit vector (g_unit) and traponse (g_unit_T)
            g_unit = np.array([[np.cos(Phis[j])*np.sin(Thetas[i])], [np.sin(Phis[j])*np.sin(Thetas[i])], [np.cos(Thetas[i])]])
            g_unit_T = np.transpose(g_unit)
            
            # DW signal for D0-3
            Sb_D0[i, j] = S0*np.exp(-b_value * np.matmul(np.matmul(g_unit_T, DT0), g_unit))
            Sb_D1[i, j] = S0*np.exp(-b_value * np.matmul(np.matmul(g_unit_T, DT1), g_unit))
            Sb_D2[i, j] = S0*np.exp(-b_value * np.matmul(np.matmul(g_unit_T, DT2), g_unit))
            Sb_D3[i, j] = S0*np.exp(-b_value * np.matmul(np.matmul(g_unit_T, DT3), g_unit))
            
            # Surface plot values for D0
            Sx_Sb_D0[i, j] = np.cos(Phis[j])*np.sin(Thetas[i])*Sb_D0[i, j]
            Sy_Sb_D0[i, j] = np.sin(Phis[j])*np.sin(Thetas[i])*Sb_D0[i, j]
            Sz_Sb_D0[i, j] = np.cos(Thetas[i])*Sb_D0[i, j]
            
            # Surface plot values for D1
            Sx_Sb_D1[i, j] = np.cos(Phis[j])*np.sin(Thetas[i])*Sb_D1[i, j]
            Sy_Sb_D1[i, j] = np.sin(Phis[j])*np.sin(Thetas[i])*Sb_D1[i, j]
            Sz_Sb_D1[i, j] = np.cos(Thetas[i])*Sb_D1[i, j]
            
            # Surface plot values for D2
            Sx_Sb_D2[i, j] = np.cos(Phis[j])*np.sin(Thetas[i])*Sb_D2[i, j]
            Sy_Sb_D2[i, j] = np.sin(Phis[j])*np.sin(Thetas[i])*Sb_D2[i, j]
            Sz_Sb_D2[i, j] = np.cos(Thetas[i])*Sb_D2[i, j]
            
            # Surface plot values for D3
            Sx_Sb_D3[i, j] = np.cos(Phis[j])*np.sin(Thetas[i])*Sb_D3[i, j]
            Sy_Sb_D3[i, j] = np.sin(Phis[j])*np.sin(Thetas[i])*Sb_D3[i, j]
            Sz_Sb_D3[i, j] = np.cos(Thetas[i])*Sb_D3[i, j]
            
            #DW signal for weighted sum of D0-3
            S_sum[i, j] = f_D0*Sb_D0[i, j] + f_D1*Sb_D1[i, j] + f_D2*Sb_D2[i, j] + f_D3*Sb_D3[i, j]
            
            # Surface plot values for weighted sum of D0-3
            Sx_S_sum[i, j] = np.cos(Phis[j])*np.sin(Thetas[i])*S_sum[i, j]
            Sy_S_sum[i, j] = np.sin(Phis[j])*np.sin(Thetas[i])*S_sum[i, j]
            Sz_S_sum[i, j] = np.cos(Thetas[i])*S_sum[i, j]
    b_value = b_value + b_value_step
    
    """
    Figure 1: Surface plot of Sx_Sb_D0, Sy_Sb_D0, and Sz_Sb_D0
    Figure 2: Surface plot of Sx_Sb_D1, Sy_Sb_D1, and Sz_Sb_D1
    Figure 3: Surface plot of Sx_Sb_D2, Sy_Sb_D2, and Sz_Sb_D2
    Figure 4: Surface plot of Sx_Sb_D3, Sy_Sb_D3, and Sz_Sb_D3
    Figure 5: Surface plot of Sx_S_Sum, Sy_S_sum, and Sz_S_sum (weighted sum of D0-3)
    """
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Sx_Sb_D0, Sy_Sb_D0, Sz_Sb_D0)
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
    if not os.path.exists(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D0\frame{}_D0.png'.format(counter_array[counter])):
        plt.savefig(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D0\frame{}_D0.png'.format(counter_array[counter]))
    
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Sx_Sb_D1, Sy_Sb_D1, Sz_Sb_D1)
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
    if not os.path.exists(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D1\frame{}_D1.png'.format(counter_array[counter])):
        plt.savefig(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D1\frame{}_D1.png'.format(counter_array[counter]))
    
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Sx_Sb_D2, Sy_Sb_D2, Sz_Sb_D2)
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
    if not os.path.exists(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D2\frame{}_D2.png'.format(counter_array[counter])):
        plt.savefig(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D2\frame{}_D2.png'.format(counter_array[counter]))
    
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Sx_Sb_D2, Sy_Sb_D2, Sz_Sb_D3)
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
    if not os.path.exists(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D3\frame{}_D3.png'.format(counter_array[counter])):
        plt.savefig(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D3\frame{}_D3.png'.format(counter_array[counter]))
    
    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Sx_S_sum, Sy_S_sum, Sz_S_sum)
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
    if not os.path.exists(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_sum\frame{}_sum.png'.format(counter_array[counter])):
        plt.savefig(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_sum\frame{}_sum.png'.format(counter_array[counter]))

    counter = counter + 1

# Create videos
os.chdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D0')
os.system("ffmpeg -framerate 10 -i frame%02d_D0.png -c:v libx264 -pix_fmt yuv420p surf_D0.mp4")

os.chdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D1')
os.system("ffmpeg -framerate 10 -i frame%02d_D1.png -c:v libx264 -pix_fmt yuv420p surf_D1.mp4")

os.chdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D2')
os.system("ffmpeg -framerate 10 -i frame%02d_D2.png -c:v libx264 -pix_fmt yuv420p surf_D2.mp4")

os.chdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_D3')
os.system("ffmpeg -framerate 10 -i frame%02d_D3.png -c:v libx264 -pix_fmt yuv420p surf_D3.mp4")

os.chdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames\frames_sum')
os.system("ffmpeg -framerate 10 -i frame%02d_sum.png -c:v libx264 -pix_fmt yuv420p surf_sum.mp4")