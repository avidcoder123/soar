import jax.numpy as jnp
from util import cl, cd
import openmdao.api as om

class AirfoilLift(om.ExplicitComponent):
    
    def setup(self):
        for x in shape_params:
                self.add_input(x)
        self.add_input("Re")
            
        self.add_output("Cl_0")
        
    def setup_partials(self):
        self.declare_partials('*', '*', method="fd")
        
    def compute(self, inputs, outputs):
        B, T, P, C, E, R = [inputs[x] for x in shape_params]
        Re = inputs["Re"]
        
        #Get Cl at zero AoA
        cl_0 = cl(B, T, P, C, E, R, jnp.float32(0), Re)

        outputs["Cl_0"] = cl_0
        
class AirfoilDrag(om.ExplicitComponent):
    
    def setup(self):
        for x in shape_params:
                self.add_input(x)
        self.add_input("Re")
            
        self.add_output("Cd")
        
    def setup_partials(self):
        self.declare_partials('*', '*', method="fd")
        
    def compute(self, inputs, outputs):
        B, T, P, C, E, R = [inputs[x] for x in shape_params]
        Re = inputs["Re"]
        
        #Get Cl at zero AoA
        cd_vmap = jax.vmap(cd, in_axes=(None, None, None, None, None, None, 0, None))
        cds = cd_vmap(B, T, P, C, E, R, alphas_eff, Re)

        outputs["Cd"] = jnp.mean(cds)