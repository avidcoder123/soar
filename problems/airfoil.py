import openmdao.api as om
from components import AirfoilLift, AirfoilDrag, EulerBernoulliBeam

def airfoil_problem(bounds, Cl_goal, initial_airfoil, alphas_eff, Re, lift_model, drag_model, b, c, v_infty, rho, fourier_names, fourier_coefficients, n_list, wing_points, youngs_modulus, metal_density, yield_strength, shear_strength, safety_factor, initial_spar, maxiter):
    
    prob = om.Problem()
        
    prob.model.add_subsystem("lift", AirfoilLift(lift_model))
    prob.model.add_subsystem("drag", AirfoilDrag(drag_model, alphas_eff))
    prob.model.add_subsystem("beam", EulerBernoulliBeam(fourier_names, n_list, wing_points, youngs_modulus, metal_density))
    
    prob.model.promotes("beam", any=["*"])
    prob.model.promotes("lift", any=["*"])
    prob.model.promotes("drag", any=["*"])
    
    for x in ["B", "T", "P", "C", "E", "R"]:
        prob.model.add_design_var(x, lower=bounds[x][0], upper=bounds[x][1])

        
    prob.model.add_design_var("web_w", lower=bounds["web_w"][0], upper=bounds["web_w"][1])
    prob.model.add_design_var("flange_w", lower=bounds["flange_w"][0], upper=bounds["flange_w"][1])
    prob.model.add_design_var("flange_h", lower=bounds["flange_h"][0], upper=bounds["flange_h"][1])
    
    prob.model.add_constraint("normal_stress", upper=yield_strength * safety_factor)
    prob.model.add_constraint("shear_stress", upper=shear_strength * safety_factor)
    prob.model.add_constraint("Cl_0", equals=Cl_goal)
    
    prob.model.add_objective("Cd")

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
    
    prob.set_val("Re", Re)
    
    prob.set_val("b", b)
    prob.set_val("c", c)
    prob.set_val("v_infty", v_infty)
    prob.set_val("rho", rho)
    
    prob.set_val("main_x", 0.25)
    prob.set_val("rear_x", 0.75)
    
    for name, An in zip(fourier_names, fourier_coefficients):
        prob.set_val(name, An)
        
    for param, val in initial_spar.items():
        prob.set_val(param, val)
    
    prob.driver.options["maxiter"] = maxiter

    prob.run_driver()

    return prob