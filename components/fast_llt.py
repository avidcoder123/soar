import jax
import jax.numpy as jnp
import openmdao.api as om
from fast_llt.aerodynamic_calculator import calculate_aerodynamics
from functools import partial

class FastLiftingLine(om.ExplicitComponent):

    def __init__(self):
        
        super().__init__()
        
    def setup(self):
        
        self.add_input("b")
        self.add_input("c")
        self.add_input("v_infty")
        self.add_input("rho")

        self.add_input("alpha_geo")
        self.add_input("alpha_0")
        self.add_input("Cd")
            
        self.add_output("L")
        self.add_output("D")
        
    def setup_partials(self):
        self.calculate_jacobian = jax.jit(jax.jacrev(self._compute_primal, argnums=range(0, 4)))
        self.declare_partials('*', '*')
        
    @partial(jax.jit, static_argnums=(0,))
    def _compute_primal(self, b, c, Cd, alpha_0, v_infty, rho, alpha_geo):
        lift, drag = calculate_aerodynamics(v_infty, rho, alpha_0, alpha_geo, c, b, Cd)
        
        return jnp.hstack([lift, drag])
        
    def compute(self, inputs, outputs):
        b = inputs["b"]
        c = inputs["c"]
        Cd = inputs["Cd"]
        
        v_infty = inputs["v_infty"]
        rho = inputs["rho"]
        alpha_geo = inputs["alpha_geo"]
        alpha_0 = inputs["alpha_0"]
        
        lift, drag = self._compute_primal(b, c, Cd, alpha_0, v_infty, rho, alpha_geo)
        outputs["L"] = lift
        outputs["D"] = drag
        
    def compute_partials(self, inputs, partials):
        b = inputs["b"]
        c = inputs["c"]
        Cd = inputs["Cd"]
        alpha_0 = inputs["alpha_0"]
        
        v_infty = inputs["v_infty"]
        rho = inputs["rho"]
        alpha_geo = inputs["alpha_geo"]
        
        jacobian = self.calculate_jacobian(b, c, Cd, alpha_0, v_infty, rho, alpha_geo)
        
        for input_name, derivative in zip(["b", "c", "Cd", "alpha_0"], jacobian):
            partials["L", input_name] = derivative[0]
            partials["D", input_name] = derivative[1]