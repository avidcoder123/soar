import jax
import jax.numpy as jnp
from util import cl, cd
import openmdao.api as om
from functools import partial

shape_params = ["B", "T", "P", "C", "E", "R"]


class AirfoilLift(om.ExplicitComponent):
    
    def __init__(self, model):
        self.model = model
        
        self.model_grad = jax.jacrev(cl, argnums=range(1, 7))
        
        super().__init__()
    
    def setup(self):
        for x in shape_params:
                self.add_input(x)
        self.add_input("Re")
            
        self.add_output("Cl_0")
        
    def setup_partials(self):
        self.declare_partials('*', '*')
        
    @partial(jax.jit, static_argnums=(0,))
    def _compute_primal(self, B, T, P, C, E, R, Re):
        return cl(self.model, B, T, P, C, E, R, jnp.float32(0), Re)
    
    @partial(jax.jit, static_argnums=(0,))
    def calculate_jacobian(self, B, T, P, C, E, R, Re):
        return self.model_grad(self.model, B, T, P, C, E, R, jnp.float32(0), Re)
        
    def compute(self, inputs, outputs):
        B, T, P, C, E, R = [inputs[x] for x in shape_params]
        Re = inputs["Re"]
        
        #Get Cl at zero AoA
        cl_0 = self._compute_primal(B, T, P, C, E, R, Re)

        outputs["Cl_0"] = cl_0
        
    def compute_partials(self, inputs, partials):
        shape_params = ["B", "T", "P", "C", "E", "R"]
        jacobian = self.calculate_jacobian(*[inputs[x] for x in shape_params + ["Re"]])
        
        for param_name, derivative in zip(shape_params, jacobian):
            partials["Cl_0", param_name] = derivative.item()
        

class AirfoilDrag(om.ExplicitComponent):
    
    def __init__(self, model, alphas_eff):
        self.model = model
        self.alphas_eff = alphas_eff
        
        cd_vmap = jax.vmap(cd, in_axes=(None, None, None, None, None, None, None, 0, None))
        self.cd_vmap = cd_vmap
                
        super().__init__()
    
    def setup(self):
        for x in shape_params:
                self.add_input(x)
        self.add_input("Re")
            
        self.add_output("Cd")
        
    def setup_partials(self):
        self.calculate_jacobian = jax.jit(jax.jacrev(self._compute_primal, argnums=range(0, 6)))
        self.declare_partials('*', '*')
        
    @partial(jax.jit, static_argnums=(0,))
    def _compute_primal(self, B, T, P, C, E, R, Re):
        cds = self.cd_vmap(self.model, B, T, P, C, E, R, self.alphas_eff, Re)
        
        return jnp.mean(cds)
        
    def compute(self, inputs, outputs):
        B, T, P, C, E, R = [inputs[x] for x in shape_params]
        Re = inputs["Re"]
        
        cds = self._compute_primal(B, T, P, C, E, R, Re)

        outputs["Cd"] = jnp.mean(cds)
        
    def compute_partials(self, inputs, partials):
        shape_params = ["B", "T", "P", "C", "E", "R"]
        jacobian = self.calculate_jacobian(*[inputs[x] for x in shape_params + ["Re"]])
        
        for param_name, derivative in zip(shape_params, jacobian):
            partials["Cd", param_name] = derivative.item()
            
# class AirfoilLift(om.ExplicitComponent):
    
#     def __init__(self, model):
#         self.model = model
        
#         super().__init__()
    
#     def setup(self):
#         for x in shape_params:
#                 self.add_input(x)
#         self.add_input("Re")
            
#         self.add_output("Cl_0")
        
#     def setup_partials(self):
#         self.declare_partials('*', '*', method="fd")
        
#     def compute(self, inputs, outputs):
#         B, T, P, C, E, R = [inputs[x] for x in shape_params]
#         Re = inputs["Re"]
        
#         #Get Cl at zero AoA
#         cl_0 = cl(self.model, B, T, P, C, E, R, jnp.float32(0), Re)

#         outputs["Cl_0"] = cl_0
            
# class AirfoilDrag(om.ExplicitComponent):
    
#     def __init__(self, model, alphas_eff):
#         self.model = model
#         self.alphas_eff = alphas_eff
        
#         super().__init__()
    
#     def setup(self):
#         for x in shape_params:
#                 self.add_input(x)
#         self.add_input("Re")
            
#         self.add_output("Cd")
        
#     def setup_partials(self):
#         self.declare_partials('*', '*', method="fd")
        
#     def compute(self, inputs, outputs):
#         B, T, P, C, E, R = [inputs[x] for x in shape_params]
#         Re = inputs["Re"]
        
#         #Get Cl at zero AoA
#         cd_vmap = jax.vmap(cd, in_axes=(None, None, None, None, None, None, None, 0, None))
#         cds = cd_vmap(self.model, B, T, P, C, E, R, self.alphas_eff, Re)

#         outputs["Cd"] = jnp.mean(cds)