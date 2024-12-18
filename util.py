import jax
import jax.numpy as jnp
from functools import partial
from surrogate import SurrogateModel
import equinox as eqx

input_labels = ["B", "T", "P", "C", "E", "R", "Alpha", "Re"]

#Generate the base network, later to be populated with weights
def generate_base_model(key=jax.random.PRNGKey(42)):
    model = SurrogateModel(
        in_size=len(input_labels),
        out_size=1,
        width_size=64,
        depth=4,
        activation=jax.nn.silu,
        key=key
    )
    
    return model

#These utility functions take multiple arguments as input rather than an array.
#They also take angle inputs in radians and convert it to degrees.
#This is because the neural network is trained in degrees.

@eqx.filter_jit
def cl(model, B, T, P, C, E, R, alpha, Re):
    alpha = jnp.rad2deg(alpha)
    return model(jnp.hstack([B, T, P, C, E, R, alpha, Re]))

@eqx.filter_jit
def cd(model, B, T, P, C, E, R, alpha, Re):
    alpha = jnp.rad2deg(alpha)
    return model(jnp.hstack([B, T, P, C, E, R, alpha, Re]))

#Get Cd but vectorize wrt theta
get_cds = jax.vmap(cd, in_axes=(None, None, None, None, None, None, None, 0, None))

#Get atmospheric variables
p0 = 101_325 #Sea level standard atmospheric pressure
cp = 1_004.68506 #Constant-pressure specific heat
g = 9.8 #Gravity
T0 = 288.16 #Sea level standard temperature	(Kelvin)
M = 0.02896968 #Molar mass of dry air
R0 = 8.314462618 #Universal gas constant
L = 6.5e-3 #Temperature lapse rate
mu0 = 1.716e-5 #Dyn viscousity at sea level
S = 110.4 #Sutherland temperature

#Get pressure from altitude
@jax.jit
def pressure(h):
    p = (g * h) / (cp * T0)
    p = 1 - p
    p = p ** ((cp * M) / R0)
    p *= p0
    
    return p

@jax.jit
def temperature(h):
    return T0 - L * h

#m/V = (PM)/(RT)
@jax.jit
def density(h):
    rho = pressure(h)
    rho *= M
    rho /= R0 * temperature(h)
    
    return rho

@jax.jit
def dyn_viscousity(h):
    mu = mu0
    T = temperature(h)
    mu *= (T / T0) ** (3/2)
    mu *= (T0 + S) / (T + S)
    
    return mu

def pretty_print(results):
    print("Airfoil")
    print("-" * 10)
    parameters = results["parameters"]
    print("Base Shape Coef. %.3f" % parameters["B"].item())
    print("Max Thickness    %.3f" % parameters["T"].item())
    print("Taper Exponent   %.3f" % parameters["P"].item())
    print("Max Camber       %.3f" % parameters["C"].item())
    print("Camber Exponent  %.3f" % parameters["E"].item())
    print("Reflex           %.3f" % parameters["R"].item())
    print("")
    
    print("Planform")
    print("-" * 10)
    print("Wingspan         %.3f m" % parameters["b"].item())
    c = parameters["c"].item()
    print("Chord length     %.3f m" % c)
    print("Aspect Ratio     %.3f" % parameters["AR"].item())
    print("")
    
    print("Spars")
    print("-" * 10)
    structure = results["structure"]
    print("Flange Height    %.3f m" % (structure["flange_h"]*c).item())
    print("Flange Width     %.3f m" % (structure["flange_w"]*c).item())
    print("Web Height       %.3f m" % (structure["web_h"]*c).item())
    print("Web Width        %.3f m" % (structure["web_w"]*c).item())
    print("Spar Ratio       %.2f" % structure["spar_ratio"].item())
    print(("Main Spar Pos.   %.2f" % structure["main_x"].item()) + "c")
    print(("Rear Spar Pos.   %.2f" % structure["rear_x"].item()) + "c")
    print("")

    
    print("Aerodynamics & Structure")
    print("-" * 10)
    aerodynamics = results["aerodynamics"]
    print("Lift             %d" % aerodynamics["L"].item())
    print("Drag             %d" % aerodynamics["D"].item())
    print("Cl at alpha=0    %.2f" % aerodynamics["Cl_0"].item())
    print("Alpha Cl=0       %.2f deg" % aerodynamics["alpha_0"].item())
    print(("Max τ threshold  %.2f" % (structure["shear"] * 100).item()) + "%")
    print(("Max σ threshold  %.2f" % (structure["normal"] * 100).item()) + "%")
    print("Material Usage   %.3f m3" % structure["material_usage"].item())
    print("")
    
    print("Timing")
    print("-" * 10)
    timing = results["timing"]
    print("Planform         %d seconds" % timing["planform"])
    print("Airfoil          %d seconds" % timing["airfoil"])
    print("Spars            %d seconds" % timing["spar"])
    print("Total            %d seconds" % (timing["planform"] + timing["airfoil"] + timing["spar"]))