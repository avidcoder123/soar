import jax
import jax.numpy as jnp

@jax.jit
def efficiency(AR):
    p = jnp.array([-1.9633426e-06,  8.4391802e-05, -1.3057156e-03,  7.9823956e-03,
        9.4408566e-01])
    
    return jnp.polyval(p, jax.nn.swish(AR))

@jax.jit
def CL_slope(AR):
    p = jnp.array([ 1.4194653e-14, -1.4275380e-13, -3.3051214e-12, -2.5009399e-11,
        2.4331867e-10,  1.0630827e-08,  1.5677905e-07,  4.0811344e-07,
       -3.5016503e-05, -7.3516683e-04,  3.5404583e-04,  2.8599551e-01,
        2.9687810e+00])
    
    return jnp.polyval(p, jax.nn.swish(AR))