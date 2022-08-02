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
import ffmpeg
import shutil
import time
from signal_calcs import *

start = time.time()

# Delete previously generated images and videos
#* CHANGE TO APPROPRIATE FOLDERS
shutil.rmtree(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_D0')
shutil.rmtree(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_D1')
shutil.rmtree(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_sum')

DT0 = dt(2.0e-3, 2.0e-3, 2.0e-3) # Rank-2 isotropic diagonal tensor, DT0 in units of mm^2/s (free water contribution)
DT =  dt(0.2e-3, 0.2e-3, 1.7e-3) # Single fiber rank-2 anisotropic diagonal tensor, DT1 in units of mm^2/s (oriented along the z-axis)
DT1 = rotate(DT, np.pi/3, np.pi/7, (5*np.pi)/4)  # Rotated DT. Angles correspond to rotation along the z-, y-, and z-axes respectively

# Compartment fractions/weightings
f_D0 = 0/2.0       # isotropic volume fraction
f_D1 = 1/2.0       # z-fiber volume fraction

# Initial signal without diffusion weighting (arbritary units)
S0 = 1.0

# Gradient unit vector directions
Thetas = np.linspace(0, np.pi, 100)   # Polar angle iterations
Phis = np.linspace(0, 2*np.pi, 100)   # Azimuthal angle iterations

# Calculate diffusion weighted signal and surface plot values

# Preallocate/initialize ararys to zeros to accelerate calculation
Sb_D0 = np.zeros((100, 100))
Sb_D1 = np.zeros((100, 100))
Sx_Sb_D0 = np.zeros((100, 100))
Sy_Sb_D0 = np.zeros((100, 100))
Sz_Sb_D0 = np.zeros((100, 100))
Sx_Sb_D1 = np.zeros((100, 100))
Sy_Sb_D1 = np.zeros((100, 100))
Sz_Sb_D1 = np.zeros((100, 100))
Sx_S_sum = np.zeros((100, 100))
Sy_S_sum = np.zeros((100, 100))
Sz_S_sum = np.zeros((100, 100))
S_sum = np.zeros((100, 100))

# Set b-value parameters
b_value = 500
b_value_stop = 3000
b_value_step = 50

# Video initialization
counter_array = ['%.2d' % a for a in range(50)] # Generate array used for labelling video frames with the number of b-values
counter = 0

#* CHANGE TO APPROPRIATE FOLDERS
os.mkdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_D0')  # Create folder for isotropic frames
os.mkdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_D1')  # Create folder for anisotropic frames
os.mkdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_sum') # Create folder for weighted sum frames

for k in range(b_value, b_value_stop, b_value_step): # Starting b-value, ending b-value, b-value increment size. All in s/mm^2
    for i in range(len(Thetas) - 1):
        for j in range(len(Phis) - 1):
            # Create gradient unit vector (g_unit) and traponse (g_unit_T)
            g_unit = np.array([[np.cos(Phis[j])*np.sin(Thetas[i])], [np.sin(Phis[j])*np.sin(Thetas[i])], [np.cos(Thetas[i])]])
            g_unit_T = np.transpose(g_unit)
            
            # DW signal
            Sb_D0[i, j] = S0*np.exp(-b_value * np.matmul(np.matmul(g_unit_T, DT0), g_unit))
            Sb_D1[i, j] = S0*np.exp(-b_value * np.matmul(np.matmul(g_unit_T, DT1), g_unit))
            
            # Surface plot values for D0
            Sx_Sb_D0[i, j] = np.cos(Phis[j])*np.sin(Thetas[i])*Sb_D0[i, j]
            Sy_Sb_D0[i, j] = np.sin(Phis[j])*np.sin(Thetas[i])*Sb_D0[i, j]
            Sz_Sb_D0[i, j] = np.cos(Thetas[i])*Sb_D0[i, j]
            
            # Surface plot values for D1
            Sx_Sb_D1[i, j] = np.cos(Phis[j])*np.sin(Thetas[i])*Sb_D1[i, j]
            Sy_Sb_D1[i, j] = np.sin(Phis[j])*np.sin(Thetas[i])*Sb_D1[i, j]
            Sz_Sb_D1[i, j] = np.cos(Thetas[i])*Sb_D1[i, j]

            #DW signal for weighted sum
            S_sum[i, j] = f_D0*Sb_D0[i, j] + f_D1*Sb_D1[i, j]
            
            # Surface plot values for weighted sum
            Sx_S_sum[i, j] = np.cos(Phis[j])*np.sin(Thetas[i])*S_sum[i, j]
            Sy_S_sum[i, j] = np.sin(Phis[j])*np.sin(Thetas[i])*S_sum[i, j]
            Sz_S_sum[i, j] = np.cos(Thetas[i])*S_sum[i, j]
    b_value = b_value + b_value_step
    
    """
    Figure 1: Surface plot of Sx_Sb_D0, Sy_Sb_D0, and Sz_Sb_D0
    Figure 2: Surface plot of Sx_Sb_D1, Sy_Sb_D1, and Sz_Sb_D1
    Figure 3: Surface plot of Sx_S_Sum, Sy_S_sum, and Sz_S_sum (weighted sum)
    """
    
    #* CHANGE TO APPROPRIATE FOLDERS
    fig = plt.figure(1)
    ax = fig.axes(projection='3d')
    ax.plot_surface(Sx_Sb_D0, Sy_Sb_D0, Sz_Sb_D0)
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    plt.savefig(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_D0\frame{}_D0.png'.format(counter_array[counter]))
    
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Sx_Sb_D1, Sy_Sb_D1, Sz_Sb_D1)
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_D1\frame{}_D1.png'.format(counter_array[counter]))
    
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Sx_S_sum, Sy_S_sum, Sz_S_sum)
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_sum\frame{}_sum.png'.format(counter_array[counter]))

    counter = counter + 1

# Create videos
# #* CHANGE TO APPROPRIATE FOLDERS
os.chdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_D0')
os.system("ffmpeg -framerate 10 -i frame%02d_D0.png -c:v libx264 -pix_fmt yuv420p surf_D0.mp4")

os.chdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_D1')
os.system("ffmpeg -framerate 10 -i frame%02d_D1.png -c:v libx264 -pix_fmt yuv420p surf_D1.mp4")

os.chdir(r'C:\Users\warrenboschen\Desktop\Free-Water-Optimization\DW_signal_frames_generalized\frames_sum')
os.system("ffmpeg -framerate 10 -i frame%02d_sum.png -c:v libx264 -pix_fmt yuv420p surf_sum.mp4")

end = time.time()
print(end - start)
# Time elapsed using PNGs as frames: 833s or ~14min