"""
Tensor Rotation
Objective: Generate a fiber population from one tensor using matrix rotations

Created 20220711 by Warren Boschen
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import dipy.sims.voxel as vox

#TODO: Try to avoid lists and loops when using numpy, as they slow down the execution speed.
"""
Define a diagonal diffusion rank-2 tensor/3x3 matrix based on its eigenvalues.

INPUTS:
xx, yy, zz (floats): The eigenvalues of the desired diagonalized matrix.

OUTPUTS:
Returns a diagonalized matrix with the appropriate eigenvalues.
"""
def dt(xx, yy, zz):
    return np.diag([xx, yy, zz])

"""
Rotate a diagonal diffusion tensor to be aligned with the Euler angles Phi, Theta, and Psi (https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix)

INPUTS:
tensor (numpy array): A rank-2 tensor/3x3 matrix.
psi (float): The first angle by which the tensor will be rotated around the z-axis. Values should be between 0 and 2pi.
theta (float): The second angle by which the tensor will be rotated around the y-axis. Values should be between 0 and 2pi.
phi (float): The third angle by which the tensor will be again rotated around the z-axis. Values should be between 0 and 2pi.

OUTPUTS:
Returns the tensor after having been rotated to the desired orientation.
"""
def rotate(tensor, psi, theta, phi):
    rot_psi = np.array([[np.cos(psi), -1*np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])           # Rotation matrix of tensor around the z-axis (occurs first)
    rot_theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1*np.sin(theta), 0, np.cos(theta)]]) # Rotation matrix of tensor around the y-axis (occurs second)
    rot_phi = np.array([[np.cos(phi), -1*np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])           # Rotation matrix of tensor around the z-axis (occurs third)
    rot1 = rot_psi @ (tensor @ np.transpose(rot_psi))   # Rotate the tensor by the angle psi.
    rot2 = rot_theta @ (rot1 @ np.transpose(rot_theta)) # Rotate the tensor by the angle theta.
    fiber = rot_phi @ (rot2 @ np.transpose(rot_phi))    # Rotate the tensor by the angle phi.
    return fiber

"""
Create an animated surface plot based on the signal profile (without noise) of a single diffusion tensor.

INPUTS:
name (string): Determines the title of the video as well as the file's name.
S0 (integer): The initial signal without diffusion weighting (arbritary units).
fiber (numpy array): A rank-2 tensor/matrix representing a WM fiber from which the signal profile will be generated.
size (integer): A generic integer used throughout the function:
    - Used when determining how many possible gradient unit vector directions there are.
    - Used when generating the zero arrays for the signal.
    - Used when looping through each of the gradient unit vectors.
eig (boolean): If true, the eigenvectors and the vectors determining the tensor's shape will be plotted along with the signal profile.

OUTPUTS:
Returns nothing.

Saves a video entitled "surf_name.mp4" depicting the the signal profile of the fiber for each diffusion weighting (b value) in blue,
plus the eigenvectors of the fiber in red and the vectors determining the fiber's shape in green if eig = True.
Note that the eigenvectors and the vectors determining the fiber's shape have been normalized and only depict the direction
of each vector, NOT their corresponding magnitude.

IDEAS:
Figure out a way to save multiple NIfTI1 images from this data set (what is Dr. Mareci looking for here?). To save a single data set, you can use
nibabel.nifti1.nifti1Image(dataobj, affine, ...), where affine likely equals np.eye(4). What shape must dataobj be?

Perhaps dataobj should be created such that there is a signal intensity corresponding to (x, y, z) for the other three dimensions. In that case, rework
the way that the data is plotted so that the signal doesn't need to be broken into its Cartesian components.
"""
def ani(name, S0, fiber, size, eig):
    Theta = np.linspace(0, np.pi, size) # Define the number of gradient unit vector directions.
    Phi = np.linspace(0, 2*np.pi, size)
    b_start = 500 # Define the starting b value, the ending b value, and the b value increment size.
    b_stop = 3000
    b_step = 50
    # Define four arrays of zeros. S is the composite DTI signal profile and Sx, Sy, and Sz are the Cartesian components of the signal.
    S = np.zeros((size, size))
    Sx = np.zeros((size, size))
    Sy = np.zeros((size, size))
    Sz = np.zeros((size, size))
    
    # Establish basic metadata about the output video.
    metadata = dict(title='Surface Plots of {}'.format(name))
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()    

    # Begin calculations and video creation.
    with writer.saving(fig, "surf_{}.mp4".format(name), size):
        for b in range(b_start, b_stop + b_step, b_step): # Starting b-value, ending b-value, b-value increment size. All in s/mm^2
            ax = fig.add_subplot(111, projection='3d') # Create the plot for the signal, including its axes and its labels.
            ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
            ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
            
            if eig:
                # Create the starting point of the eigenvectors and the vectors defining the fiber's shape at the origin.
                x = [0, 0, 0]
                y = [0, 0, 0]
                z = [0, 0, 0]
                eigenvalues, eigenvectors = np.linalg.eig(fiber) # Find the eigenvalues/vectors of the fiber and assign them to a Cartesian component.
                x_eig = [row[0] for row in eigenvectors]
                y_eig = [row[1] for row in eigenvectors]
                z_eig = [row[2] for row in eigenvectors]
                ax.quiver(x, y, z, x_eig, y_eig, z_eig, color='r', length = 0.7, normalize = True) # Plot the normalized eigenvectors.
                x_vec = [100*row[0] for row in fiber]
                y_vec = [100*row[1] for row in fiber]
                z_vec = [100*row[2] for row in fiber]
                ax.quiver(x, y, z, x_vec, y_vec, z_vec, color='g', length = 0.7, normalize = True) # Plot the normalized vectors defining the tensor's shape.

            for i in range(size): # For each possible unit vector along the gradient direction...
                for j in range(size):
                    # Create gradient unit vector (g_unit) and transpose (g_unit_T).
                    g_unit = np.array([[np.cos(Phi[j])*np.sin(Theta[i])], [np.sin(Phi[j])*np.sin(Theta[i])], [np.cos(Theta[i])]])
                    g_unit_T = np.transpose(g_unit)
                    
                    # Calculate the DW signal.
                    S[i, j] = S0*np.exp(-b * np.matmul(np.matmul(g_unit_T, fiber), g_unit))
                    
                    # Split the DW signal into its components for plotting.
                    Sx[i, j] = np.cos(Phi[j])*np.sin(Theta[i])*S[i, j]
                    Sy[i, j] = np.sin(Phi[j])*np.sin(Theta[i])*S[i, j]
                    Sz[i, j] = np.cos(Theta[i])*S[i, j]
            ax.plot_surface(Sx, Sy, Sz) #* Add color='' if a specific color is desired.
            writer.grab_frame() # Save the entire plot as a frame of the video "surf_name.mp4".

"""
Create an animated surface plot based on the signal profile (with noise) of a single diffusion tensor.

INPUTS:
name (string): Determines the title of the video as well as the file's name.
S0 (integer): The initial signal without diffusion weighting (arbritary units).
fiber (numpy array): A rank-2 tensor/matrix representing a WM fiber from which the signal profile will be generated.
size (integer): A generic integer used throughout the function:
    - Used when determining how many possible gradient unit vector directions there are.
    - Used when generating the zero arrays for the signal.
    - Used when looping through each of the gradient unit vectors.
eig (boolean): If true, the eigenvectors and the vectors determining the tensor's shape will be plotted along with the signal profile.

OUTPUTS:
Returns nothing.

Saves a video entitled "surf_name.mp4" depicting the the signal profile of the fiber for each diffusion weighting (b value) in blue,
plus the eigenvectors of the fiber in red and the vectors determining the fiber's shape in green if eig = True.
Note that the eigenvectors and the vectors determining the fiber's shape have been normalized and only depict the direction
of each vector, NOT their corresponding magnitude. Unlike the function ani, this function includes Rician distributed noise.

TODO: Even out the noise a little bit more.
#* See the function add_noise here: https://github.com/dipy/dipy/blob/master/dipy/sims/phantom.py.
#! Comments with an exclamation point denote additions and changes from the function ani.
"""
def noise_test(name, S0, fiber, size, eig):
    Theta = np.linspace(0, np.pi, size) # Define the number of gradient unit vector directions and create Cartesian axes.
    Phi = np.linspace(0, 2*np.pi, size)
    THETA, PHI = np.meshgrid(Theta, Phi)
    X = np.sin(THETA) * np.cos(PHI) #!
    Y = np.sin(THETA) * np.sin(PHI) #!
    Z = np.cos(THETA) #!
    b_start = 500 # Define the starting b value, the ending b value, and the b value increment size.
    b_stop = 3000
    b_step = 50
    # Define four arrays of zeros. S is the composite DTI signal profile and Sx, Sy, and Sz are the Cartesian components of the signal. #!
    S = np.array([X, Y, Z, np.zeros((size, size))])
    Sx = np.zeros((size, size))
    Sy = np.zeros((size, size))
    Sz = np.zeros((size, size))
    
    # Establish basic metadata about the output video.
    metadata = dict(title='Surface Plots of {}'.format(name))
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()    

    # Begin calculations and video creation.
    with writer.saving(fig, "surf_{}.mp4".format(name), size):
        for b in range(b_start, b_stop + b_step, b_step): # Starting b-value, ending b-value, b-value increment size. All in s/mm^2
            ax = fig.add_subplot(111, projection='3d') # Create the plot for the signal, including its axes and its labels.
            ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))
            ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
            
            if eig:
                # Create the starting point of the eigenvectors and the vectors defining the fiber's shape at the origin.
                x = [0, 0, 0]
                y = [0, 0, 0]
                z = [0, 0, 0]
                eigenvalues, eigenvectors = np.linalg.eig(fiber) # Find the eigenvalues/vectors of the fiber and assign them to a Cartesian component.
                x_eig = [row[0] for row in eigenvectors]
                y_eig = [row[1] for row in eigenvectors]
                z_eig = [row[2] for row in eigenvectors]
                ax.quiver(x, y, z, x_eig, y_eig, z_eig, color='r', length = 0.7, normalize = True) # Plot the normalized eigenvectors.
                x_vec = [100*row[0] for row in fiber]
                y_vec = [100*row[1] for row in fiber]
                z_vec = [100*row[2] for row in fiber]
                ax.quiver(x, y, z, x_vec, y_vec, z_vec, color='g', length = 0.7, normalize = True) # Plot the normalized vectors defining the tensor's shape.

            for i in range(size): # For each possible unit vector along the gradient direction...
                for j in range(size):
                    # Create gradient unit vector (g_unit) and transpose (g_unit_T).
                    g_unit = np.array([[np.cos(Phi[j])*np.sin(Theta[i])], [np.sin(Phi[j])*np.sin(Theta[i])], [np.cos(Theta[i])]])
                    g_unit_T = np.transpose(g_unit)
                    
                    # Calculate the DW signal.
                    S[3, i, j] = S0*np.exp(-b * np.matmul(np.matmul(g_unit_T, fiber), g_unit)) #!
            
            S = vox.add_noise(S, 2.0, 0.5) # Add noise to the calculated signal matrix. #!
            
            # Split the DW signal into its components for plotting.
            for i in range(size):
                for j in range(size):
                    Sx[i, j] = np.cos(Phi[j])*np.sin(Theta[i])*S[3, i, j]
                    Sy[i, j] = np.sin(Phi[j])*np.sin(Theta[i])*S[3, i, j]
                    Sz[i, j] = np.cos(Theta[i])*S[3, i, j]
            ax.plot_surface(Sx, Sy, Sz) #* Add color='' if a specific color is desired.
            writer.grab_frame() # Save the entire plot as a frame of the video "surf_name.mp4".

"""
Create an animated surface plot based on the signal profile of a single diffusion tensor.

INPUTS:
S0 (integer): The initial signal without diffusion weighting (arbritary units).
fibers (list): A Python list of rank-2 tensors/3x3 matrices representing some number of WM fibers from which the signal profile will be generated.
weightings (list): A Python list of floats of the fiber weightings. The first fiber corresponds to the first weighting, etc.
size (integer): A generic integer used throughout the function:
    - Used when determining how many possible gradient unit vector directions there are.
    - Used when generating the zero arrays for the signal.
    - Used when looping through each of the gradient unit vectors.

OUTPUTS:
Returns nothing.

Saves a video entitled "surf_sum.mp4" depicting the eigenvectors of the fiber in red, the vectors determining the fiber's shape in green,
and the signal profile of the fiber for each diffusion weighting (b value) in blue. Note that the eigenvectors and the vectors determining the fiber's
shape have been normalized and only depict the direction of each vector, NOT their corresponding magnitude.

TODO: Figure out how to display the eigenvectors and the vectors defining the tensor.
"""
def ani_multi(S0, fibers, weightings, size):
    Thetas = np.linspace(0, np.pi, size) # Define the number of gradient unit vector directions.
    Phis = np.linspace(0, 2*np.pi, size)
    fibers = np.array(fibers)
    weightings = np.array(weightings)
    b_start = 500 # Define the starting b value, the ending b value, and the b value increment size.
    b_stop = 3000
    b_step = 50
    S = np.zeros((size, size)) # Define four arrays of zeros. S is the composite DTI signal profile and Sx, Sy, and Sz are the Cartesian components of the signal.
    Sx = np.zeros((size, size))
    Sy = np.zeros((size, size))
    Sz = np.zeros((size, size))
    
    # Establish basic metadata about the output video.
    metadata = dict(title='Surface Plots of Weighted Sum')
    writer = FFMpegWriter(fps=10, metadata=metadata)
    fig = plt.figure()    

    # Begin calculations and video creation.
    with writer.saving(fig, "surf_sum.mp4", size):
        for b in range(b_start, b_stop + b_step, b_step): # Starting b-value, ending b-value, b-value increment size. All in s/mm^2
            ax = fig.add_subplot(111, projection='3d')
            ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
            ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
            for f in range(len(fibers)): # For each fiber...
                for i in range(size): # And for each possible unit vector along the gradient direction...
                    for j in range(size):
                        # Create gradient unit vector (g_unit) and transpose (g_unit_T).
                        g_unit = np.array([[np.cos(Phis[j])*np.sin(Thetas[i])], [np.sin(Phis[j])*np.sin(Thetas[i])], [np.cos(Thetas[i])]])
                        g_unit_T = np.transpose(g_unit)
                        
                        # Calculate the DW signal.
                        S[i, j] += S0*weightings[f]*np.exp(-b * np.matmul(np.matmul(g_unit_T, fibers[f]), g_unit))
                        
                        # Split the DW signal into its components for plotting.
                        Sx[i, j] += np.cos(Phis[j])*np.sin(Thetas[i])*S[i, j]
                        Sy[i, j] += np.sin(Phis[j])*np.sin(Thetas[i])*S[i, j]
                        Sz[i, j] += np.cos(Thetas[i])*S[i, j]
            ax.plot_surface(Sx, Sy, Sz)
            writer.grab_frame() # Save the entire plot as a frame of the video "surf_name.mp4".
            S = np.zeros((size, size)) # Reset the signal back to 0 for the next b value.
            Sx = np.zeros((size, size))
            Sy = np.zeros((size, size))
            Sz = np.zeros((size, size))
