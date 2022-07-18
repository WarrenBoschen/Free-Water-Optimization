"""
Tensor Rotation
Objective: Generate a fiber population from one tensor using matrix rotations

Created 20220711 by Warren Boschen
"""

from stat import ST_SIZE
import numpy as np
import matplotlib
from sympy import re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# Define a diagonal diffusion tensor based on its eigenvalues
def dt(xx, yy, zz):
    return np.diag([xx, yy, zz])

# Rotate a diagonal diffusion tensor to be aligned with the Euler angles Phi, Theta, and Psi (https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix) 
def rotate(tensor, psi, theta, phi):
    rot_psi = np.array([[np.cos(psi), -1*np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])           # Rotation matrix of tensor around the z-axis (occurs first)
    rot_theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1*np.sin(theta), 0, np.cos(theta)]]) # Rotation matrix of tensor around the y-axis (occurs second)
    rot_phi = np.array([[np.cos(phi), -1*np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])           # Rotation matrix of tensor around the z-axis (occurs third)
    rot = rot_phi @ rot_theta @ rot_psi  # Multiply each of the rotation matrices together
    rot_T = np.transpose(rot)    # Transpose the rotation matrix
    fiber = rot @ tensor @ rot_T # Find rotated tensor
    return fiber

# Create an animated surface plot based on the signal profile of a single diffusion tensor.
def ani(name, S0, fiber, size):
    Thetas = np.linspace(0, np.pi, size)
    Phis = np.linspace(0, 2*np.pi, size)
    b_start = 500
    b_stop = 3000
    b_step = 50
    S = np.zeros((size, size))
    Sx = np.zeros((size, size))
    Sy = np.zeros((size, size))
    Sz = np.zeros((size, size))
    
    metadata = dict(title='Surface Plots of {}'.format(name))
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()    

    with writer.saving(fig, "surf_{}.mp4".format(name), size):
        for b in range(b_start, b_stop + b_step, b_step): # Starting b-value, ending b-value, b-value increment size. All in s/mm^2
            ax = fig.add_subplot(111, projection='3d')
            ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
            ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
            for i in range(size):
                for j in range(size):
                    # Create gradient unit vector (g_unit) and transpose (g_unit_T)
                    g_unit = np.array([[np.cos(Phis[j])*np.sin(Thetas[i])], [np.sin(Phis[j])*np.sin(Thetas[i])], [np.cos(Thetas[i])]])
                    g_unit_T = np.transpose(g_unit)
                    
                    # DW signal
                    S[i, j] = S0*np.exp(-b * np.matmul(np.matmul(g_unit_T, fiber), g_unit))
                    
                    # Surface plot values
                    Sx[i, j] = np.cos(Phis[j])*np.sin(Thetas[i])*S[i, j]
                    Sy[i, j] = np.sin(Phis[j])*np.sin(Thetas[i])*S[i, j]
                    Sz[i, j] = np.cos(Thetas[i])*S[i, j]
            ax.plot_surface(Sx, Sy, Sz)
            writer.grab_frame()

def ani_multi(S0, fibers, weightings, size):
    Thetas = np.linspace(0, np.pi, size)
    Phis = np.linspace(0, 2*np.pi, size)
    fibers = np.array(fibers)
    weightings = np.array(weightings)
    b_start = 500
    b_stop = 3000
    b_step = 50
    S = np.zeros((size, size))
    Sx = np.zeros((size, size))
    Sy = np.zeros((size, size))
    Sz = np.zeros((size, size))
    
    metadata = dict(title='Surface Plots of Weighted Sum')
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()    

    with writer.saving(fig, "surf_sum.mp4", size):
        for b in range(b_start, b_stop + b_step, b_step): # Starting b-value, ending b-value, b-value increment size. All in s/mm^2
            ax = fig.add_subplot(111, projection='3d')
            ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
            ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
            for f in range(len(fibers)):
                for i in range(size):
                    for j in range(size):
                        # Create gradient unit vector (g_unit) and transpose (g_unit_T)
                        g_unit = np.array([[np.cos(Phis[j])*np.sin(Thetas[i])], [np.sin(Phis[j])*np.sin(Thetas[i])], [np.cos(Thetas[i])]])
                        g_unit_T = np.transpose(g_unit)
                        
                        # DW signal
                        S[i, j] += S0*weightings[f]*np.exp(-b * np.matmul(np.matmul(g_unit_T, fibers[f]), g_unit))
                        
                        # Surface plot values
                        Sx[i, j] += np.cos(Phis[j])*np.sin(Thetas[i])*S[i, j]
                        Sy[i, j] += np.sin(Phis[j])*np.sin(Thetas[i])*S[i, j]
                        Sz[i, j] += np.cos(Thetas[i])*S[i, j]
            ax.plot_surface(Sx, Sy, Sz)
            writer.grab_frame()
