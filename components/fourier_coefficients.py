import openmdao.api as om
from lifting_line.fourier_solver import solve_coefficients

#Fourier coefficients of the circulation distribution
class FourierCoefficients(om.ExplicitComponent):
    
    def __init__(self, fourier_names, n_list, wing_points):
        self.fourier_names = fourier_names
        self.n_list = n_list
        self.wing_points = wing_points
        
        super().__init__()
    
    def setup(self):
        
        self.add_input("b")
        self.add_input("c")
        self.add_input("alpha_0")
        self.add_input("v_infty")
        self.add_input("rho")
        self.add_input("mu")
        self.add_input("alpha_geo")
        
        for x in self.fourier_names:
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
        
        coefficients = solve_coefficients(self.n_list, self.wing_points, alpha_geo, c, b, alpha_0)
        
        for name, value in zip(self.fourier_names, coefficients):
            outputs[name] = value