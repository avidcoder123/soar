# SOAR: Efficient Wing Shape Optimization using Auto-Differentiable Neural Network Surrogate Models

Wing shape optimization is crucial for creating efficient aircraft, yet contemporary approaches utilizing Computational Fluid Dynamics (CFD) simulations with the adjoint method suffer from poor performance and incompatibility with modern automatic differentiation techniques.
This project aims to address these shortcomings by developing a novel, computationally efficient wing shape optimizer.

Neural networks are leveraged to predict the lift and drag polars of a two-dimensional airfoil based on its parameterized shape.

Lifting-line theory is used to generalize the airfoil polars to a finite wing.
This is implemented by representing the wing circulation distribution as a Fourier series, simplifying the fundamental equation of lifting-line theory and significantly improving solver performance.

For structural optimization, we focus on optimizing the wing spars, which we represent as two I-beams. The deflection and stresses along the wing are approximated using Euler-Bernoulli beam theory. Numerical integration is utilized to efficiently find an approximate solution.

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
Holds the files defining the optimization problems.

### planform.py
The planform problem focuses on optimizing the wingspan and chord length of the wing. However, there is also some basic spar optimization to ensure that the wing will be structurally realistic.

### airfoil.py
Focuses on optimizing the airfoil shape to match the $C_{L,\alpha=0}$ goal and minimize drag at $\alpha_\mathrm{eff}$. There is also basic spar optimization to ensure that the airfoil shape is structurally sound.

### spar.py
Optimizes the spar dimensions and positions to minimize material usage while making sure the stresses are within safe limits.

## /components
Contains modules calculating different variables used in the optimizer.

### airfoil_surrogates.py
Calculates airfoil lift and drag polars given a parameterized airfoil.

### beam.py
Uses Euler-Bernoulli beam theory to analyze the normal and shear stresses on the wing based on its geometry and the lift distribution.

### fourier_coefficients.py
Solves the Fourier coefficients representing the circulation distribution of the wing using a gradient-based nonlinear solver.

### lift0.py
A simple utility component converting $C_{L,\alpha=0}$ to $\alpha_{L=0}$.

### lifting_line.py
Uses the calculated Fourier coefficients and wing geometry to calculate the lift and drag force.

### reynolds_calculator.py
A simple utility component calculating the Reynolds number.

## /lifting_line
Functions to perform lifting-line calculations

### aerodynamic_calculator.py
Takes the Fourier coefficients representing the circulation distribution and uses integration to find the total lift and drag force on the wing.

### fourier_solver.py
Solves the system of nonlinear equations of the Fourier Series reformulation of lifting-line theory.

### fourier_util.py
Contains functions to calculate circulation and $\alpha_i$ from Fourier coefficients.

## /eb_beam
Functions to perform Euler-Bernoulli beam calculations

### skin.py
Calculates the area, 1st moment of area $Q$, and 2nd moment of area $I$ of the wing skin.

### solve_beam.py
Calculates the beam's shear, moment, slope, and deflection distribution using numerical cumulative integration.

### spar.py
Calculates the area, 1st moment of area $Q$, and 2nd moment of area $I$ of the spar I-beams.

### stress.py
Uses shear and moment distirbution to calculate the maximum shear and normal stress.

## /notebooks
Contains Jupyter notebooks not directly used in the codebase, but rather for analysis and data generation.

### liftcurve.ipynb
A "sandbox" to graph the lift curve of an airfoil based on shape parameters. Used during development of the airfoil surrogate models.

### scrape.ipynb
Scrapes airfoil coordinates and polars from online databases and fits the coordinates to 6 shape parameters.

## /validation
Contains tests to evaluate the effectiveness of the optimizer system.

### designspace_search.ipynb
Used to find the design space for the airfoils. Evaluates which airfoil shapes obey the rule of thin-airfoil theory stating $\frac{\mathrm{d}C_l}{\mathrm{d}\alpha}\approx 2\pi$.

### gridsearch.ipynb
Performs a grid search on possible hyperparameters for the lifting-line solver.

### gridsearch_results.ipynb
Ranks the results of the grid search.

### optim_evaluation.ipynb
Runs the optimizer for a variety of lift goals and flight conditions to test its capabilities.
