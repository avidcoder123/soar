import openmdao.api as om
from components import AirfoilLift, AirfoilDrag

def airfoil_problem(bounds, Cl_goal, initial_airfoil, Re, maxiter):
    prob = om.Problem()
    
    prob.model.add_subsystem("lift", AirfoilLift())
    prob.model.add_subsystem("drag", AirfoilDrag())

    prob.model.promotes("lift", any=["*"])
    prob.model.promotes("drag", any=["*"])
    
    for x in ["B", "T", "P", "C", "E", "R"]:
        prob.model.add_design_var(x, lower=bounds[x][0], upper=bounds[x][1])

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
    
    prob.driver.options["maxiter"] = maxiter

    prob.run_driver()

    return prob