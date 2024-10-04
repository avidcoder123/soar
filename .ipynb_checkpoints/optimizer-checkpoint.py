import jax
import jax.numpy as jnp
import equinox as eqx
from util import generate_base_model

class Optimizer():
    
    #Parameters to adjust the discretization
    def __init__(self, num_coefficients=10, wing_points=25):
        self.num_coefficients = num_coefficients
        self.wing_points = wing_points
        
        #Subscripts for the circulation Fourier series
        #(Only odd numbers)
        self.n_list = jnp.arange(1, self.num_coefficients * 2 + 1, 2)
    
    #Starting airfoil values
    B = 1.79094660282135
    T = 0.1879087835550308
    P = 2.998518228530884
    C = 0.0500964857637882
    E = 0.9923266768455504
    R = 0.0028237155638635
    
    def set_initial_airfoil(self, B, T, P, C, E, R):
        self.B = B
        self.T = T
        self.P = P
        self.C = C
        self.E = E
        self.R = R
        
    #Bounds for the design variables
    dv_bounds = {
        "B": (1.5, 1.8),
        "T": (0.15, 0.25),
        "P": (2.5, 3.5),
        "C": (0, 0.75),
        "E": (0.6, 1),
        "R": (-0.02, 0.02),
        "c": (1, 4),
        "AR": (5, 15)
    }
    
    def set_dv_bounds(self, name, bounds):
        if getattr(bounds, "__len__", None) is not None and len(bounds) == 2:
            self.dv_bounds[name] = bounds
        else:
            raise ValueError("Bounds must be a tuple with length 2.")
            
    def load_surrogates(self, lift_file, drag_file):
        base_model = generate_base_model()
        
        self.lift_surrogate = eqx.tree_deserialise_leaves(lift_file, base_model)
        self.drag_surrogate = eqx.tree_deserialise_leaves(drag_file, base_model)