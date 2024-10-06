import jax.numpy as jnp
import openmdao.api as om

class Lift0(om.ExplicitComponent):
    
    def setup(self):
        self.add_input("Cl_0")
        
        self.add_output("alpha_0")
        
    def setup_partials(self):
        self.declare_partials('*', '*', method="fd")
        
    def compute(self, inputs, outputs):
        cl_0 = inputs["Cl_0"]

        #Using thin airfoil theory, assume that Cl is a straight
        #line with slope=2pi and find the x-intercept
        alpha_0 = -cl_0/(2 * jnp.pi)

        outputs["alpha_0"] = alpha_0