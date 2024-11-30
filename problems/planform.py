import openmdao.api as om
from components import FourierCoefficients, Lift0, LiftingLine, ReynoldsCalculator, EulerBernoulliBeam
from util import cl
import jax.numpy as jnp

def planform_problem(bounds, fourier_names, n_list, wing_points, lift_goal, initial_airfoil, v_infty, mu, rho, alpha_geo, youngs_modulus, metal_density, yield_strength, shear_strength, safety_factor, lift_model, drag_model, tolerance, maxiter):
    prob = om.Problem()
    
    B = initial_airfoil["B"]
    T = initial_airfoil["T"]
    P = initial_airfoil["P"]
    C = initial_airfoil["C"]
    E = initial_airfoil["E"]
    R = initial_airfoil["R"]
    

    prob.model.add_subsystem("lift_0", Lift0())
    prob.model.add_subsystem("reynolds", ReynoldsCalculator())
    prob.model.add_subsystem("circulation", FourierCoefficients(fourier_names, n_list, wing_points))
    prob.model.add_subsystem("llt", LiftingLine(fourier_names, n_list, wing_points, drag_model))
    #TODO: Make material properties adjustable
    prob.model.add_subsystem("beam", EulerBernoulliBeam(fourier_names, n_list, wing_points, youngs_modulus, metal_density))
    prob.model.add_subsystem("aspect_ratio", om.ExecComp("AR = b / c"))

    prob.model.promotes("lift_0", any=["*"])
    prob.model.promotes("reynolds", any=["*"])
    prob.model.promotes("circulation", any=["*"])
    prob.model.promotes("llt", any=["*"])
    prob.model.promotes("aspect_ratio", any=["*"])
    prob.model.promotes("beam", any=["*"])

    prob.model.add_design_var("c", lower=bounds["c"][0], upper=bounds["c"][1])
    prob.model.add_design_var("b", lower=bounds["c"][0] * bounds["AR"][0], upper=bounds["c"][1] * bounds["AR"][1])

    prob.model.add_design_var("Cl_0", lower=0)

    prob.model.add_constraint("AR", lower=bounds["AR"][0], upper=bounds["AR"][1])
    prob.model.add_constraint("L", equals=lift_goal)
    
    prob.model.add_constraint("normal_stress", upper=yield_strength * safety_factor)
    prob.model.add_constraint("shear_stress", upper=shear_strength * safety_factor)
    
    prob.model.add_objective("D")

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.setup()

    prob.set_val("B", B)
    prob.set_val("T", T)
    prob.set_val("P", P)
    prob.set_val("C", C)
    prob.set_val("E", E)
    prob.set_val("R", R)

    prob.set_val("c", 3)
    prob.set_val("b", 30)
    
    Cl_0 = cl(lift_model, B, T, P, C, E, R, jnp.float32(0), 1_000_000)
    prob.set_val("Cl_0", Cl_0)

    prob.set_val("v_infty", v_infty)
    prob.set_val("mu", mu)
    prob.set_val("rho", rho)
    prob.set_val("alpha_geo", alpha_geo)

    prob.driver.options["tol"] = tolerance
    prob.driver.options["maxiter"] = maxiter
    
    prob.run_driver()
    return prob