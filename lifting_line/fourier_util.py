import jax
import jax.numpy as jnp

#Calculate circulation from Fourier coefficients
@jax.jit
def circulation_fn(theta, coefficients, n_list, b, v_infty):
    gamma = 2 * b * v_infty
    summation_fn = jax.vmap(lambda An, n: An * jnp.sin(n * theta))
    gamma *= summation_fn(coefficients, n_list).sum()
    
    return gamma

#Calculate induced angle of attack from Fourier coefficients
@jax.jit
def alpha_i_fn(theta, coefficients, n_list):
    summation_fn = jax.vmap(lambda An, n: n * An * jnp.sin(n * theta) / jnp.sin(theta))
    radians = summation_fn(coefficients, n_list).sum()   
    return radians