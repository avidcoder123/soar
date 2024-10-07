import jax
import jax.numpy as jnp
import equinox as eqx
from util import generate_base_model
from problems import planform_problem, airfoil_problem
from lifting_line.fourier_util import alpha_i_fn

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
        "C": (0, 0.75),
        "E": (0.6, 1),
        "R": (-0.02, 0.02),
        "c": (1, 4),
        "AR": (5, 15)
    }
    
    def set_dv_bounds(self, name, bounds):
        if getattr(bounds, "__len__", None) is not None and len(bounds) == 2:
            self.dv_bounds[name] = bounds
        else:
            raise ValueError("Bounds must be a tuple with length 2.")
            
    def load_surrogates(self, lift_model, drag_model):
        
        self.lift_surrogate = lift_model
        self.drag_surrogate = drag_model
        
    def solve_wing(self, lift_goal, v_infty, mu, rho, alpha_geo, lift_tolerance=1e3, maxiter=50):
        if getattr(self, "lift_surrogate", None) is None or getattr(self, "drag_surrogate", None) is None:
            raise ValueError("Surrogate models were not initialized. Did you forget to call load_surrogates?")
        
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
            lift_model=self.lift_surrogate,
            drag_model=self.drag_surrogate,
            tolerance=lift_tolerance,
            maxiter=maxiter
        )
        
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
        
        prob = airfoil_problem(
            bounds=self.dv_bounds,
            Cl_goal=Cl_0,
            initial_airfoil=self.initial_airfoil,
            alphas_eff=alphas_eff,
            Re=Re,
            lift_model=self.lift_surrogate,
            drag_model=self.drag_surrogate,
            maxiter=maxiter
        )
        
        optimized_airfoil = dict()
        
        for x in ["B", "T", "P", "C", "E", "R"]:
            optimized_airfoil[x] = prob.get_val(x)
        
        #Return the ideal parameters
        return {
            **optimized_airfoil,
            **optimized_planform
        }