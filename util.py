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