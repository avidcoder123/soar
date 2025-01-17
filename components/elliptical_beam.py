import jax
import jax.numpy as jnp
import openmdao.api as om
from eb_beam.skin import skin_moments
from eb_beam.spar import spar_moments, thickness_from_x
from eb_beam.solve_beam import solve_beam
from eb_beam.stress import max_stress
from functools import partial

class EllipticalEulerBernoulliBeam(om.ExplicitComponent):
    
    def __init__(self, wing_points, youngs_modulus, metal_density):
        self.wing_points = wing_points
        self.metal_density = metal_density
        self.youngs_modulus = youngs_modulus
        
        
        super().__init__()
        
    def setup(self):
        
        self.add_input("b")
        for x in ["B", "T", "P", "C", "E", "R"]:
            self.add_input(x)
    
        self.add_input("L")