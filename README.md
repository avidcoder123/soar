# SOAR: Efficient Wing Shape Optimization using Auto-Differentiable Neural Network Surrogate Models

Wing shape optimization is crucial for creating efficient aircraft, yet contemporary approaches utilizing Computational Fluid Dynamics (CFD) simulations with the adjoint method suffer from poor performance and incompatibility with modern automatic differentiation techniques.
This project aims to address these shortcomings by developing a novel, computationally efficient wing shape optimizer.

Neural networks are leveraged to predict the lift and drag polars of a two-dimensional airfoil based on its parameterized shape.
Lifting-line theory is used to generalize the airfoil polars to a finite wing.
This is implemented by representing the wing circulation distribution as a Fourier series, simplifying the fundamental equation of lifting-line theory and significantly improving solver performance.
The optimizer is implemented using OpenMDAO with the SLSQP optimization algorithm.

# Folder and file structure
## Root folder
### optimizer.py
This file is the main entry point for usage of the optimizer. It provides functions to alter default values and bounds, set up surrogate models, define the optimization objectives, and run the optimizer.

### surrogate.py
Holds the Equinox class defining the neural network for airfoil polars. Note that the lift and drag networks are seperate but have the same structure.

### util.py
Contains functions to interact with the neural networks and helper functions for atmospheric conditions.
