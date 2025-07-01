import jax
import jax.numpy as jnp
from util import cd as cd_fn
from .surrogates import efficiency, CL_slope
import equinox as eqx


@eqx.filter_jit
def calculate_aerodynamics(v_infty, rho, alpha_0, alpha_geo, c, b, Cd):
    AR = b/c
    
    e = efficiency(AR)
    CL = CL_slope(AR) * (alpha_geo - alpha_0)
    
    CDi = CL ** 2 / (jnp.pi * AR * e)
    CD = Cd + CDi
    
    q = 0.5 * rho * v_infty ** 2
    S = b * c
    
    lift = CL * q * S
    drag = CD * q * S
    
    return jnp.array([lift, drag])