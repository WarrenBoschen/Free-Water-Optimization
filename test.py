
import numpy as np
size = 100
Theta = np.linspace(0, np.pi, size) # Define the number of gradient unit vector directions.
Phi = np.linspace(0, 2*np.pi, size)
THETA, PHI = np.meshgrid(Theta, Phi)
X = np.sin(THETA) * np.cos(PHI)
print(X.shape)
Y = np.sin(THETA) * np.sin(PHI)
print(Y.shape)
Z = np.cos(THETA)
print(Z.shape)