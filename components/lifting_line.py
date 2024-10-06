from jax.numpy import jnp
from openmdao.api import om
from lifting_line.aerodynamic_calculator import calculate_aerodynamics

class LiftingLine(om.ExplicitComponent):
    
    def setup(self, fourier_names, n_list):
        self.n_list = n_list
        
        self.add_input("b")
        self.add_input("c")
        self.add_input("Re")
        self.add_input("v_infty")
        self.add_input("rho")
        self.add_input("mu")
        for x in ["B", "T", "P", "C", "E", "R"]:
            self.add_input(x)
        for x in fourier_names:
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
        B, T, P, C, E, R = [inputs[x] for x in shape_params]
        coefficients = jnp.array([inputs[x] for x in fourier_names])
        
        lift, drag = calculate_aerodynamics(coefficients, self.n_list, v_infty, rho, Re, alpha_geo, c, b, B, T, P, C, E, R)
        outputs["L"] = lift
        outputs["D"] = drag