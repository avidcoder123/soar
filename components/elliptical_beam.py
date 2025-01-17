import jax
import jax.numpy as jnp
import openmdao.api as om
from eb_beam.skin import skin_moments
from eb_beam.spar import spar_moments, thickness_from_x
from eb_beam.solve_elliptical import solve_beam
from eb_beam.stress import max_stress
from functools import partial

class EllipticalEulerBernoulliBeam(om.ExplicitComponent):
    
    def __init__(self, youngs_modulus, metal_density):
        self.metal_density = metal_density
        self.youngs_modulus = youngs_modulus
        
        
        super().__init__()
        
    def setup(self):
        
        self.add_input("b")
        self.add_input("c")
        self.add_input("alpha_geo")
        self.add_input("L")
        
        for x in ["B", "T", "P", "C", "E", "R"]:
            self.add_input(x)
            
        self.add_input("web_w")
        self.add_input("flange_w")
        self.add_input("flange_h")
        self.add_input("main_x")
        self.add_input("rear_x")
        
        self.add_output("normal_stress")
        self.add_output("shear_stress")
        self.add_output("material_usage")
        self.add_output("main_overflow")
        self.add_output("rear_overflow")
        self.add_output("spar_distance")
        
    def setup_partials(self):
        self.declare_partials('*', '*', method="fd")
            
    @partial(jax.jit, static_argnums=(0,))
    def _compute_primal(self, b, c, L, B, T, P, C, E, R, alpha_geo, main_web_w, main_flange_w, main_flange_h, main_x, rear_x):
        shape_params = [B, T, P, C, E, R]
        
        #Preset skin thickness
        t = 0.05
        
        skin_I, skin_Q0, skin_A = skin_moments(t, T, c, alpha_geo)
        
        main_I, main_Q0, main_A = spar_moments(*shape_params, c, main_web_w, main_flange_w, main_flange_h, main_x, alpha_geo)
        
        
        ratio = thickness_from_x(rear_x, *shape_params[:3]) / thickness_from_x(main_x, *shape_params[:3])

        rear_web_w = main_web_w * ratio
        rear_flange_w = main_flange_w * ratio
        rear_flange_h = main_flange_h * ratio
        
        rear_I, rear_Q0, rear_A = spar_moments(*shape_params, c, rear_web_w, rear_flange_w, rear_flange_h, rear_x, alpha_geo)
        
        I = skin_I + main_I + rear_I
        Q0 = skin_Q0 + main_Q0 + rear_Q0
        A = skin_I + main_A + rear_A
        
        weight_loading = A * self.metal_density * 9.81
        
        #Get shear and moment
        V, M = solve_beam(L, b)
        
        #Integrate weight loading
        V -= (b/2) * weight_loading
        M -= 0.5 * (b/2) ** 2 * weight_loading
        
        E = self.youngs_modulus
        normal_stress, shear_stress = max_stress(V, M, T, t, c, main_web_w, rear_web_w, I, Q0)
        
        material_usage = A * b
        
        main_overflow = main_x - main_flange_w/2
        rear_overflow = rear_x + rear_flange_w/2 - 1
        spar_distance = (rear_x - rear_flange_w/2) - (main_x + main_flange_w/2)
        
        #The signs are different because the main overflow is to the left and rear overflow is to the right
        main_overflow = -main_overflow
        
        return jnp.hstack([normal_stress, shear_stress, material_usage, main_overflow, rear_overflow, spar_distance])
    
    def compute(self, inputs, outputs):
        b = inputs["b"]
        c = inputs["c"]
        L = inputs["L"]
        alpha_geo = inputs["alpha_geo"]
        shape_params = [inputs[x] for x in ["B", "T", "P", "C", "E", "R"]]
        web_w = inputs["web_w"]
        flange_w = inputs["flange_w"]
        flange_h = inputs["flange_h"]
        main_x = inputs["main_x"]
        rear_x = inputs["rear_x"]
        
        normal_stress, shear_stress, material_usage, main_overflow, rear_overflow, spar_distance = self._compute_primal(b, c, L, *shape_params, alpha_geo, web_w, flange_w, flange_h, main_x, rear_x)
        
        outputs["normal_stress"] = normal_stress
        outputs["shear_stress"] = shear_stress
        outputs["material_usage"] = material_usage
        
        outputs["main_overflow"] = main_overflow
        outputs["rear_overflow"] = rear_overflow
        outputs["spar_distance"] = spar_distance