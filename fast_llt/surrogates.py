import jax
import jax.numpy as jnp

@jax.jit
def efficiency(AR):
    p = jnp.array([-1.9633426e-06,  8.4391802e-05, -1.3057156e-03,  7.9823956e-03,
        9.4408566e-01])
    
    return jnp.polyval(p, jax.nn.swish(AR))

@jax.jit
def CL_slope(AR):
    p = jnp.array([-1.6805009e-06,  6.0792572e-05, -1.2155521e-03,  8.9416429e-03,
        4.5288625e+00])
    
    return jnp.polyval(p, jax.nn.swish(AR))