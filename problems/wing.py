import openmdao.api as om
from components import Lift0, FastLiftingLine, ReynoldsCalculator, EllipticalEulerBernoulliBeam, AirfoilLift, AirfoilDrag
from util import cl
import jax.numpy as jnp
def wing_problem(bounds, lift_goal, safety_factor, initial_airfoil, v_infty, mu, rho, alpha_geo, lift_model, drag_model, tolerance, youngs_modulus, metal_density, yield_strength, shear_strength, maxiter):
    prob = om.Problem()
    
   
    prob.model.add_subsystem("lift", AirfoilLift(lift_model))
    prob.model.add_subsystem("drag", AirfoilDrag(drag_model, alpha_geo))
    prob.model.add_subsystem("lift_0", Lift0())
    prob.model.add_subsystem("reynolds", ReynoldsCalculator())
    prob.model.add_subsystem("llt", FastLiftingLine())


    prob.model.add_subsystem("beam", EllipticalEulerBernoulliBeam(youngs_modulus, metal_density))
    prob.model.add_subsystem("aspect_ratio", om.ExecComp("AR = b / c"))

    prob.model.promotes("lift_0", any=["*"])
    prob.model.promotes("reynolds", any=["*"])
    prob.model.promotes("llt", any=["*"])
    prob.model.promotes("aspect_ratio", any=["*"])
    prob.model.promotes("lift", any=["*"])
    prob.model.promotes("drag", any=["*"])
    prob.model.promotes("beam", any=["*"])

    prob.model.add_design_var("c", lower=bounds["c"][0], upper=bounds["c"][1])
    prob.model.add_design_var("b", lower=bounds["c"][0] * bounds["AR"][0], upper=bounds["c"][1] * bounds["AR"][1])
    
    for x in ["B", "T", "P", "C", "E", "R"]:
        prob.model.add_design_var(x, lower=bounds[x][0], upper=bounds[x][1])
    prob.model.add_design_var("web_w", lower=bounds["web_w"][0], upper=bounds["web_w"][1])
    prob.model.add_design_var("flange_w", lower=bounds["flange_w"][0], upper=bounds["flange_w"][1])
    prob.model.add_design_var("flange_h", lower=bounds["flange_h"][0], upper=bounds["flange_h"][1])

    prob.model.add_constraint("AR", lower=bounds["AR"][0], upper=bounds["AR"][1])
    prob.model.add_constraint("L", equals=lift_goal)
    
    prob.model.add_constraint("normal_stress", upper=yield_strength * safety_factor)
    prob.model.add_constraint("shear_stress", upper=shear_strength * safety_factor)
    #Min alpha_0 is at -10 deg
    prob.model.add_constraint("alpha_0", upper=0, lower=jnp.deg2rad(-10))
    
    prob.model.add_objective("D")

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.setup()
    
    B = initial_airfoil["B"]
    T = initial_airfoil["T"]
    P = initial_airfoil["P"]
    C = initial_airfoil["C"]
    E = initial_airfoil["E"]
    R = initial_airfoil["R"]

    prob.set_val("B", B)
    prob.set_val("T", T)
    prob.set_val("P", P)
    prob.set_val("C", C)
    prob.set_val("E", E)
    prob.set_val("R", R)

    prob.set_val("c", 2)
    prob.set_val("b", 20)
    prob.set_val("web_w", 0.01)
    prob.set_val("flange_w", 0.1)
    prob.set_val("flange_h", 0.025)
    prob.set_val("main_x", 0.25)
    prob.set_val("rear_x", 0.75)

    prob.set_val("v_infty", v_infty)
    prob.set_val("mu", mu)
    prob.set_val("rho", rho)
    prob.set_val("alpha_geo", alpha_geo)

    prob.driver.options["tol"] = tolerance
    prob.driver.options["maxiter"] = maxiter
    
    prob.run_driver()
    return prob