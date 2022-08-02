# Free-Water-Optimization

The purpose of these functions is to simulate the signal profile of a voxel with multiple fiber populations represented as tensors. The free-water component is measured as an isotropic tensor, and white matter (WM) fibers are represented as anistropic tensors.

Below is a brief summary of the roles of each file in this repository:

- dw_ani.py: The main file in which calculations are performed.
- signal_calcs.py: A collection of functions made for general use.
    dt: Generates a diagonal tensor from three inputs.
    rotate: Rotates a tensor to be aligned with the input angles Phi, Theta, and Psi.

DEFUNCT/UNUSED:
- dw_signal.py: Creates five movies depicting the signal profile of the free-water component, three orthogonal WM fibers, and a weighted sum of the previous four. This is accomplished by repeatedly creating PNGs and using those as the frames of the movies.
- dw_signal_generalized.py: Creates three movies depicting the signal profile of the free-water component, a WM fiber with a particular orientation, and a weighted sum of the previous two. This is accomplished by repeatedly creating PNGs and using those as the frames of the movies.
- dw_signal_generalized_ani.py: Creates three movies depicting the signal profile of the free-water component, a WM fiber with a particular orientation, and a weighted sum of the previous two. This is accomplished by using the FFMpegWriter feature from the matplotlib.animation module.
