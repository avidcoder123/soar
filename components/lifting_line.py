import jax.numpy as jnp
import openmdao.api as om
from lifting_line.aerodynamic_calculator import calculate_aerodynamics

class LiftingLine(om.ExplicitComponent):
    
    def __init__(self, fourier_names, n_list, wing_points, drag_model):
        self.fourier_names = fourier_names
        self.n_list = n_list
        self.drag_model = drag_model
        self.wing_points = wing_points
        
        super().__init__()
    
    def setup(self):
        
        self.add_input("b")
        self.add_input("c")
        self.add_input("Re")
        self.add_input("v_infty")
        self.add_input("rho")
        self.add_input("mu")
        for x in ["B", "T", "P", "C", "E", "R"]:
            self.add_input(x)
        for x in self.fourier_names:
            self.add_input(x)
        self.add_input("alpha_geo")
            
        self.add_output("L")
        self.add_output("D")
        
        
    def setup_partials(self):
        self.declare_partials('*', '*', method="fd")
        
    def compute(self, inputs, outputs):
        b = inputs["b"]
        c = inputs["c"]
        
        v_infty = inputs["v_infty"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        alpha_geo = inputs["alpha_geo"]
        
        Re = inputs["Re"]
        shape_params = [inputs[x] for x in ["B", "T", "P", "C", "E", "R"]]
        coefficients = jnp.array([inputs[x] for x in self.fourier_names])
        
        lift, drag = calculate_aerodynamics(self.drag_model, coefficients, self.n_list, self.wing_points, v_infty, rho, Re, alpha_geo, c, b, *shape_params)
        outputs["L"] = lift
        outputs["D"] = drag