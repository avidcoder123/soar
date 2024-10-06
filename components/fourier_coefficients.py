from openmdao.api import om
from lifting_line.fourier_solver import solve_coefficients

#Fourier coefficients of the circulation distribution
class FourierCoefficients(om.ExplicitComponent):
    
    def setup(self, fourier_names, n_list):
        self.n_list = n_list
        
        self.add_input("b")
        self.add_input("c")
        self.add_input("alpha_0")
        self.add_input("v_infty")
        self.add_input("rho")
        self.add_input("mu")
        self.add_input("alpha_geo")
        
        for x in fourier_names:
            self.add_output(x)
            
    def setup_partials(self):
        self.declare_partials('*', '*', method="fd")
        
    def compute(self, inputs, outputs):
        b = inputs["b"]
        c = inputs["c"]
        
        v_infty = inputs["v_infty"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        alpha_geo = inputs["alpha_geo"]
        
        alpha_0 = inputs["alpha_0"]
        
        coefficients = solve_coefficients(self.n_list, v_infty, rho, mu, alpha_geo, c, b, alpha_0)
        
        for name, value in zip(fourier_names, coefficients):
            outputs[name] = value