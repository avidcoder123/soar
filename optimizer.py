import jax
import jax.numpy as jnp
import equinox as eqx
from util import generate_base_model
from problems import planform_problem, airfoil_problem, spar_problem
from lifting_line.fourier_util import alpha_i_fn
from lifting_line.aerodynamic_calculator import calculate_aerodynamics
from eb_beam.spar import thickness_from_x
import time
import openmdao.api as om
import warnings

class Optimizer():
    
    #Parameters to adjust the discretization
    def __init__(self, num_coefficients=10, wing_points=25):
        self.num_coefficients = num_coefficients
        self.wing_points = wing_points
        
        #Subscripts for the circulation Fourier series
        #(Only odd numbers)
        self.n_list = jnp.arange(1, self.num_coefficients * 2 + 1, 2)
        self.fourier_names = [f"A_{n}" for n in self.n_list]
    
    #Starting airfoil values
    initial_airfoil = {
        "B": 1.79094660282135,
        "T": 0.1879087835550308,
        "P": 2.998518228530884,
        "C": 0.0500964857637882,
        "E": 0.9923266768455504,
        "R": 0.0028237155638635   
    }
    
    def set_initial_airfoil(self, B, T, P, C, E, R):
        self.initial_airfoil["B"] = B
        self.initial_airfoil["T"] = T
        self.initial_airfoil["P"] = P
        self.initial_airfoil["C"] = C
        self.initial_airfoil["E"] = E
        self.initial_airfoil["R"] = R
        
    #Bounds for the design variables
    dv_bounds = {
        "B": (1.5, 1.8),
        "T": (0.15, 0.25),
        "P": (2.5, 3.5),
        "C": (0, 0.175),
        "E": (0.6, 1),
        "R": (-0.02, 0.02),
        "c": (1, 6),
        "AR": (5, 15),
        "web_w": (0.005, 0.1),
        "flange_w": (0.1, 0.25),
        "flange_h": (0.005, 0.1),
        "main_x": (0.05, 0.45),
        "rear_x": (0.55, 0.95)
    }
    
    def set_dv_bounds(self, name, bounds):
        if getattr(bounds, "__len__", None) is not None and len(bounds) == 2:
            self.dv_bounds[name] = bounds
        else:
            raise ValueError("Bounds must be a tuple with length 2.")
            
    def load_surrogates(self, lift_model, drag_model):
        
        self.lift_surrogate = lift_model
        self.drag_surrogate = drag_model
        
    def define_material(self, E, rho, yield_strength, shear_strength):
        self.material = dict()
        self.material["youngs_modulus"] = E
        self.material["metal_density"] = rho
        self.material["yield_strength"] = yield_strength
        self.material["shear_strength"] = shear_strength
        
    def solve_wing(self, lift_goal, safety_factor, v_infty, mu, rho, alpha_geo, tol=0.01, maxiter=100):

        if getattr(self, "lift_surrogate", None) is None or getattr(self, "drag_surrogate", None) is None:
            raise ValueError("Surrogate models were not initialized. Did you forget to call load_surrogates?")
            
        if getattr(self, "material", None) is None:
            raise ValueError("Wing material not initialized. Did you forget to call define_material?")
            
        #Within 1% of lift goal
        lift_tolerance = lift_goal * tol
        
        print("Optimizing planform")
        planform_time = time.time()
        prob = planform_problem(
            bounds=self.dv_bounds,
            fourier_names=self.fourier_names,
            n_list=self.n_list,
            wing_points=self.wing_points,
            lift_goal=lift_goal,
            initial_airfoil=self.initial_airfoil,
            v_infty=v_infty,
            mu=mu,
            rho=rho,
            alpha_geo=alpha_geo,
            youngs_modulus=self.material["youngs_modulus"],
            metal_density=self.material["metal_density"],
            yield_strength=self.material["yield_strength"],
            shear_strength=self.material["shear_strength"],
            safety_factor=safety_factor,
            lift_model=self.lift_surrogate,
            drag_model=self.drag_surrogate,
            tolerance=lift_tolerance,
            maxiter=maxiter
        )
        planform_time = time.time() - planform_time
        
        flange_w = prob.get_val("flange_w")
        flange_h = prob.get_val("flange_h")
        web_w = prob.get_val("web_w")
        
        #Get planform design variables
        optimized_planform = dict()
        for x in ["b", "c"]:
            optimized_planform[x] = prob.get_val(x)

        #Values to pass on to next problem
        Cl_0 = prob.get_val("Cl_0")
        Re = prob.get_val("Re")
        
        #Get the effective alphas to calculate drag for
        fourier_coefficients = jnp.hstack([prob.get_val(x) for x in self.fourier_names])
        thetas = jnp.linspace(1e-3, jnp.pi - 1e-3, self.wing_points)

        alphas_i = jax.vmap(alpha_i_fn, in_axes=(0, None, None))(thetas, fourier_coefficients, self.n_list)
        alphas_eff = alpha_geo - alphas_i     
        
        print("Optimizing airfoil")
        airfoil_time = time.time()
        prob = airfoil_problem(
            bounds=self.dv_bounds,
            Cl_goal=Cl_0,
            initial_airfoil=self.initial_airfoil,
            alphas_eff=alphas_eff,
            Re=Re,
            lift_model=self.lift_surrogate,
            drag_model=self.drag_surrogate,
            b=optimized_planform["b"],
            c=optimized_planform["c"],
            v_infty=v_infty,
            rho=rho,
            fourier_names=self.fourier_names,
            fourier_coefficients=fourier_coefficients,
            n_list=self.n_list,
            wing_points=self.wing_points,
            youngs_modulus=self.material["youngs_modulus"],
            metal_density=self.material["metal_density"],
            yield_strength=self.material["yield_strength"],
            shear_strength=self.material["shear_strength"],
            safety_factor=safety_factor,
            initial_spar={
                "web_w": web_w,
                "flange_w": flange_w,
                "flange_h": flange_h
            },
            maxiter=maxiter * 5
        )
        airfoil_time = time.time() - airfoil_time
        
        optimized_airfoil = dict()
        for x in ["B", "T", "P", "C", "E", "R"]:
            optimized_airfoil[x] = prob.get_val(x)
            
        flange_w = prob.get_val("flange_w")
        flange_h = prob.get_val("flange_h")
        web_w = prob.get_val("web_w")
            
        lift, drag = calculate_aerodynamics(
            drag_model=self.drag_surrogate,
            coefficients=fourier_coefficients,
            n_list=self.n_list,
            wing_points=1000,
            v_infty=v_infty,
            rho=rho,
            Re=Re,
            alpha_geo=alpha_geo,
            **optimized_planform,
            **optimized_airfoil
        )
        
        print("Optimizing spars")
        spar_time = time.time()
        prob = spar_problem(
            bounds=self.dv_bounds,
            b=optimized_planform["b"],
            c=optimized_planform["c"],
            v_infty=v_infty,
            rho=rho,
            airfoil=optimized_airfoil,
            fourier_names=self.fourier_names,
            fourier_coefficients=fourier_coefficients,
            n_list=self.n_list,
            wing_points=self.wing_points,
            youngs_modulus=self.material["youngs_modulus"],
            metal_density=self.material["metal_density"],
            yield_strength=self.material["yield_strength"],
            shear_strength=self.material["shear_strength"],
            safety_factor=safety_factor,
            initial_spar={
                "web_w": web_w,
                "flange_w": flange_w,
                "flange_h": flange_h
            },
            maxiter=maxiter
        )
        spar_time = time.time() - spar_time
        
        main_x = prob.get_val("main_x")
        rear_x = prob.get_val("rear_x")
        
        normal_stress = prob.get_val("normal_stress")
        shear_stress = prob.get_val("shear_stress")
        
        flange_w = prob.get_val("flange_w")
        flange_h = prob.get_val("flange_h")
        web_w = prob.get_val("web_w")
        
        material_usage = prob.get_val("material_usage")
        
        main_web_h = thickness_from_x(main_x, optimized_airfoil["B"], optimized_airfoil["T"], optimized_airfoil["P"])
        rear_web_h = thickness_from_x(rear_x, optimized_airfoil["B"], optimized_airfoil["T"], optimized_airfoil["P"])

        
        #Return the ideal parameters
        return {
            "parameters": {
                **optimized_airfoil,
                **optimized_planform,
                "AR": optimized_planform["b"]/optimized_planform["c"]
            },
            "aerodynamics": {
                "L": lift,
                "D": drag,
                "Cl_0": Cl_0,
                "alpha_0": jnp.rad2deg(-Cl_0/(2 * jnp.pi))
            },
            "structure": {
                "normal": normal_stress / self.material["yield_strength"],
                "shear": shear_stress / self.material["shear_strength"],
                "flange_w": flange_w,
                "flange_h": flange_h,
                "web_w": web_w,
                "web_h": main_web_h,
                "spar_ratio": main_web_h / rear_web_h,
                "main_x": main_x,
                "rear_x": rear_x,
                "material_usage": material_usage
            },
            "timing": {
                "planform": planform_time,
                "airfoil": airfoil_time,
                "spar": spar_time
            }
        }