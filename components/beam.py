import jax
import jax.numpy as jnp
import openmdao.api as om
from eb_beam.skin import skin_moments
from eb_beam.spar import spar_moments, thickness_from_x
from eb_beam.solve_beam import solve_beam
from eb_beam.stress import max_stress
from functools import partial

# class EulerBernoulliBeam(om.ExplicitComponent):
    
#     def __init__(self, fourier_names, n_list, wing_points, youngs_modulus, metal_density):
#         self.fourier_names = fourier_names
#         self.n_list = n_list
#         self.wing_points = wing_points
#         self.metal_density = metal_density
#         self.youngs_modulus = youngs_modulus
        
#         super().__init__()
    
#     def setup(self):
        
#         self.add_input("b")
#         self.add_input("c")
#         self.add_input("v_infty")
#         self.add_input("rho")
#         for x in ["B", "T", "P", "C", "E", "R"]:
#             self.add_input(x)
#         for x in self.fourier_names:
#             self.add_input(x)
#         self.add_input("alpha_geo")
#         self.add_input("web_w")
#         self.add_input("flange_w")
#         self.add_input("flange_h")
#         self.add_input("main_x")
#         self.add_input("rear_x")
        
#         self.add_output("normal_stress")
#         self.add_output("shear_stress")
#         self.add_output("material_usage")
#         self.add_output("main_overflow")
#         self.add_output("rear_overflow")
#         self.add_output("spar_distance")
        
#     def setup_partials(self):
#         self.declare_partials('*', '*', method="fd")
        
#     def compute(self, inputs, outputs):
#         thetas = jnp.linspace(jnp.pi/2, jnp.pi, self.wing_points)
#         summation_fn = jax.vmap(lambda An, n: An * jnp.sin(n * thetas))
#         b = inputs["b"]
        
#         coefficients = jnp.array([inputs[x] for x in self.fourier_names])
#         gammas = summation_fn(coefficients, self.n_list).sum(axis=0)
#         q = gammas * 2 * b * inputs["rho"] * inputs["v_infty"] ** 2
        
#         c = inputs["c"]
#         shape_params = [inputs[x] for x in ["B", "T", "P", "C", "E", "R"]]
#         T = shape_params[1]
#         alpha_geo = inputs["alpha_geo"]
        
#         #Preset skin thickness
#         t = 0.05
        
#         skin_I, skin_Q0, skin_A = skin_moments(t, T, c, alpha_geo)
        
#         #Preset spar sizing
#         #Main Spar
#         main_x = inputs["main_x"]#0.25 #Main spar at 0.25c

#         # main_web_w = 0.01
#         # main_flange_w = 0.1
#         # main_flange_h = 0.025
#         main_web_w = inputs["web_w"]
#         main_flange_w = inputs["flange_w"]
#         main_flange_h = inputs["flange_h"]
        
#         main_I, main_Q0, main_A = spar_moments(*shape_params, c, main_web_w, main_flange_w, main_flange_h, main_x, alpha_geo)
        
        
#         #Rear Spar
#         rear_x = inputs["rear_x"] #0.75 #Rear spar at 0.75c

#         ratio = thickness_from_x(rear_x, *shape_params[:3]) / thickness_from_x(main_x, *shape_params[:3])

#         rear_web_w = main_web_w * ratio
#         rear_flange_w = main_flange_w * ratio
#         rear_flange_h = main_flange_h * ratio
        
#         rear_I, rear_Q0, rear_A = spar_moments(*shape_params, c, rear_web_w, rear_flange_w, rear_flange_h, rear_x, alpha_geo)
        
#         I = skin_I + main_I + rear_I
#         Q0 = skin_Q0 + main_Q0 + rear_Q0
#         A = skin_I + main_A + rear_A
        
#         #Weight of metal in N/m
#         weight_loading = A * self.metal_density * 9.81
#         q += weight_loading
        
#         E = self.youngs_modulus
#         X1 = -(b/2) * jnp.cos(thetas)
#         y, _, M, V = solve_beam(q, E*I, X1)
                
#         normal_stress, shear_stress = max_stress(V, M, T, t, c, main_web_w, rear_web_w, I, Q0)
        
#         outputs["normal_stress"] = normal_stress
#         outputs["shear_stress"] = shear_stress
#         outputs["material_usage"] = A * b
        
#         main_overflow = main_x - main_flange_w/2
#         rear_overflow = rear_x + rear_flange_w/2 - 1
#         spar_distance = (rear_x - rear_flange_w/2) - (main_x + main_flange_w/2)
        
#         #The signs are different because the main overflow is to the left and rear overflow is to the right
#         outputs["main_overflow"] = -main_overflow
#         outputs["rear_overflow"] = rear_overflow
#         outputs["spar_distance"] = spar_distance

class EulerBernoulliBeam(om.ExplicitComponent):
    
    def __init__(self, fourier_names, n_list, wing_points, youngs_modulus, metal_density):
        self.fourier_names = fourier_names
        self.n_list = n_list
        self.wing_points = wing_points
        self.metal_density = metal_density
        self.youngs_modulus = youngs_modulus
        
        
        super().__init__()
    
    def setup(self):
        
        self.add_input("b")
        self.add_input("c")
        self.add_input("v_infty")
        self.add_input("rho")
        self.add_input("alpha_geo")
        for x in ["B", "T", "P", "C", "E", "R"]:
            self.add_input(x)
        for x in self.fourier_names:
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
        
        self.fourier_jacobian = jax.jacrev(self._compute_primal, argnums=(5,))
        self.spar_jacobian = jax.jacrev(self._compute_primal, argnums=range(6, 17))
        
    def setup_partials(self):
        self.declare_partials(of=["normal_stress", "shear_stress", "material_usage", "main_overflow", "rear_overflow", "spar_distance"], wrt=["B", "T", "P", "C", "E", "R", "web_w", "flange_w", "flange_h", "main_x", "rear_x"] + self.fourier_names)

        
    @partial(jax.jit, static_argnums=(0,))
    def _compute_primal(self, b, c, v_infty, rho, alpha_geo, coefficients, B, T, P, C, E, R, main_web_w, main_flange_w, main_flange_h, main_x, rear_x):
        thetas = jnp.linspace(jnp.pi/2, jnp.pi, self.wing_points)
        summation_fn = jax.vmap(lambda An, n: An * jnp.sin(n * thetas))
        
        gammas = summation_fn(coefficients, self.n_list).sum(axis=0)
        q = gammas * 2 * b * rho * v_infty ** 2
        
        shape_params = [B, T, P, C, E, R]
        T = shape_params[1]
        
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
        
        #Weight of metal in N/m
        weight_loading = A * self.metal_density * 9.81
        q += weight_loading
        
        E = self.youngs_modulus
        X1 = -(b/2) * jnp.cos(thetas)
        y, _, M, V = solve_beam(q, E*I, X1)
                
        normal_stress, shear_stress = max_stress(V, M, T, t, c, main_web_w, rear_web_w, I, Q0)
        
        material_usage = A * b
        
        main_overflow = main_x - main_flange_w/2
        rear_overflow = rear_x + rear_flange_w/2 - 1
        spar_distance = (rear_x - rear_flange_w/2) - (main_x + main_flange_w/2)
        
        #The signs are different because the main overflow is to the left and rear overflow is to the right
        main_overflow = -main_overflow

        return jnp.hstack([normal_stress, shear_stress, material_usage, main_overflow, rear_overflow, spar_distance])
    
    @partial(jax.jit, static_argnums=(0,))
    def _calculate_jacobian(self, b, c, v_infty, rho, alpha_geo, coefficients, B, T, P, C, E, R, web_w, flange_w, flange_h, main_x, rear_x):
        fourier_jacobian = self.fourier_jacobian(b, c, v_infty, rho, alpha_geo, coefficients, B, T, P, C, E, R, web_w, flange_w, flange_h, main_x, rear_x)
        fourier_jacobian = fourier_jacobian[0].reshape(6, 10)
        
        spar_jacobian = self.spar_jacobian(b, c, v_infty, rho, alpha_geo, coefficients, B, T, P, C, E, R, web_w, flange_w, flange_h, main_x, rear_x)
        spar_jacobian = jnp.hstack(spar_jacobian)
        
        return (fourier_jacobian, spar_jacobian)
        
    def compute(self, inputs, outputs):
        b = inputs["b"]
        c = inputs["c"]
        v_infty = inputs["v_infty"]
        rho = inputs["rho"]
        alpha_geo = inputs["alpha_geo"]
        coefficients = jnp.array([inputs[x] for x in self.fourier_names])
        shape_params = [inputs[x] for x in ["B", "T", "P", "C", "E", "R"]]
        web_w = inputs["web_w"]
        flange_w = inputs["flange_w"]
        flange_h = inputs["flange_h"]
        main_x = inputs["main_x"]
        rear_x = inputs["rear_x"]
        
        (normal_stress, shear_stress, material_usage, main_overflow, rear_overflow, spar_distance) = self._compute_primal(b, c, v_infty, rho, alpha_geo, coefficients, *shape_params, web_w, flange_w, flange_h, main_x, rear_x)
        
        
        outputs["normal_stress"] = normal_stress
        outputs["shear_stress"] = shear_stress
        outputs["material_usage"] = material_usage
        
        #The signs are different because the main overflow is to the left and rear overflow is to the right
        outputs["main_overflow"] = main_overflow
        outputs["rear_overflow"] = rear_overflow
        outputs["spar_distance"] = spar_distance
        
    def compute_partials(self, inputs, partials):
        b = inputs["b"]
        c = inputs["c"]
        v_infty = inputs["v_infty"]
        rho = inputs["rho"]
        alpha_geo = inputs["alpha_geo"]
        coefficients = jnp.array([inputs[x] for x in self.fourier_names])
        shape_params = [inputs[x] for x in ["B", "T", "P", "C", "E", "R"]]
        web_w = inputs["web_w"]
        flange_w = inputs["flange_w"]
        flange_h = inputs["flange_h"]
        main_x = inputs["main_x"]
        rear_x = inputs["rear_x"]
    
        fourier_jacobian, spar_jacobian = self._calculate_jacobian(b, c, v_infty, rho, alpha_geo, coefficients, *shape_params, web_w, flange_w, flange_h, main_x, rear_x)
                                        
        for func_name, (fourier_row, spar_row) in zip(["normal_stress", "shear_stress", "material_usage", "main_overflow", "rear_overflow", "spar_distance"], zip(fourier_jacobian, spar_jacobian)):
            
            #Fourier coefficients only have an impact on stress
            for fourier_name, derivative in zip(self.fourier_names, fourier_row):
                partials[func_name, fourier_name] = derivative
                
            for spar_name, derivative in zip(["B", "T", "P", "C", "E", "R", "web_w", "flange_w", "flange_h", "main_x", "rear_x"], spar_row):
                partials[func_name, spar_name] = derivative