import openmdao.api as om
from components import EulerBernoulliBeam

def spar_problem(bounds, b, c, v_infty, rho, airfoil, fourier_names, fourier_coefficients, n_list, wing_points, youngs_modulus, metal_density, yield_strength, shear_strength, safety_factor, maxiter):
    
    prob = om.Problem()
    
    prob.model.add_subsystem("beam", EulerBernoulliBeam(fourier_names, n_list, wing_points, youngs_modulus, metal_density))
    prob.model.promotes("beam", any=["*"])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.setup()
    
    prob.model.add_design_var("web_w", lower=bounds["web_w"][0], upper=bounds["web_w"][1])
    prob.model.add_design_var("flange_w", lower=bounds["flange_w"][0], upper=bounds["flange_w"][1])
    prob.model.add_design_var("flange_h", lower=bounds["flange_h"][0], upper=bounds["flange_h"][1])

    prob.set_val(b, "b")
    prob.set_val(c, "c")
    prob.set_val(v_infty, "v_infty")
    prob.set_val(rho, "rho")
    
    for x in ["B", "T", "P", "C", "E", "R"]:
        prob.set_val(x, airfoil[x])
    
    for name, An in zip(fourier_names, fourier_coefficients):
        prob.set_val(name, An)
        
    prob.driver.options["maxiter"] = maxiter

    prob.run_driver()

    return prob