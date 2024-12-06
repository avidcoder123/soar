# SOAR: Efficient Wing Shape Optimization using Auto-Differentiable Neural Network Surrogate Models

Wing shape optimization is crucial for creating efficient aircraft, yet contemporary approaches utilizing Computational Fluid Dynamics (CFD) simulations with the adjoint method suffer from poor performance and incompatibility with modern automatic differentiation techniques.
This project aims to address these shortcomings by developing a novel, computationally efficient wing shape optimizer.

Neural networks are leveraged to predict the lift and drag polars of a two-dimensional airfoil based on its parameterized shape.

Lifting-line theory is used to generalize the airfoil polars to a finite wing.
This is implemented by representing the wing circulation distribution as a Fourier series, simplifying the fundamental equation of lifting-line theory and significantly improving solver performance.

For structural optimization, we focus on optimizing the wing spars, which we represent as two I-beams. The deflection and stresses along the wing are approximated using Euler-Bernoillo beam theory. Numerical integration is utilized to efficiently find an approximate solution.

The optimizer is implemented using OpenMDAO with the SLSQP optimization algorithm.

# Folder and file structure
## Root folder
### optimizer.py
This file is the main entry point for usage of the optimizer. It provides functions to alter default values and bounds, set up surrogate models, define the optimization objectives, and run the optimizer.

### surrogate.py
Holds the Equinox class defining the neural network for airfoil polars. Note that the lift and drag networks are seperate but have the same structure.

### util.py
Contains functions to interact with the neural networks and helper functions for atmospheric conditions.

### main.ipynb
A Jupyter Notebook showing an example of how the optimizer may be used.

## /problems
Holds the files defining the optimization problems

### planform.py
The planform problem focuses on optimizing the wingspan and chord length of the wing. However, there is also some basic spar optimization to ensure that the wing will be structurally realistic.

### airfoil.py
Focuses on optimizing the airfoil shape to match the $C_{L,\alpha=0} goal and minimize drag at $\alpha_\mathrm{eff}$
