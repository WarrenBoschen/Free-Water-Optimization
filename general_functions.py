"""
Tensor Rotation
Objective: Generate a fiber population from one tensor using matrix rotations

Created 20220711 by Warren Boschen
"""

import numpy as np
import matplotlib.pyplot as plt

# Define a diagonal diffusion tensor based on its eigenvalues
def dt(xx, yy, zz):
    return np.diag([xx, yy, zz])

# Rotate a diagonal diffusion tensor to be aligned with the Euler angles Phi, Theta, and Psi (https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix) 
def rotate(tensor, phi, theta, psi):
    rot_phi = np.array([[np.cos(phi), -1*np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])           # Rotation matrix of tensor around the z-axis (occurs third)
    rot_theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1*np.sin(theta), 0, np.cos(theta)]]) # Rotation matrix of tensor around the y-axis (occurs second)
    rot_psi = np.array([[np.cos(psi), -1*np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])           # Rotation matrix of tensor around the z-axis (occurs first)
    rot = rot_phi @ rot_theta @ rot_psi  # Multiply each of the rotation matrices together
    rot_T = np.transpose(rot)    # Transpose the rotation matrix
    fiber = rot @ tensor @ rot_T # Find rotated tensor
    return fiber