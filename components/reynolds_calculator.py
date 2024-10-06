import openmdao.api as om

class ReynoldsCalculator(om.ExplicitComponent):
    
    def setup(self):
        self.add_input("c")
        self.add_input("v_infty")
        self.add_input("rho")
        self.add_input("mu")
        
        self.add_output("Re")
        
    def setup_partials(self):
        self.declare_partials('*', '*', method="fd")
        
    def compute(self, inputs, outputs):
        c = inputs["c"]
        v_infty = inputs["v_infty"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        
        outputs["Re"] = (rho * v_infty * c)/mu