import jax
import jax.numpy as jnp

@jax.jit
def cumulative_trapezoid(y, x, axis=-1):
    dx_array = jnp.moveaxis(jnp.diff(x, axis=axis), axis, -1)
    y_arr = jnp.moveaxis(y, axis, -1)
    
    
    return 0.5 * jnp.cumsum(dx_array * (y_arr[..., 1:] + y_arr[..., :-1]))

#q is the load distribution in N/m at several points along the wing
#EI is the flexural rigidity (Young's Modulus * 2nd Moment of Area)
#X1 is the length of the beam (length of a single wing)
@jax.jit
def solve_beam(q, EI, X1):
    V = cumulative_trapezoid(y=q, x=X1)
    V = jnp.insert(V, 0, 0) - V[-1] #Boundary conditions
    
    M = cumulative_trapezoid(y=V, x=X1)
    M = jnp.insert(M, 0, 0) - M[-1] #Boundary conditions
    
    m = cumulative_trapezoid(y=M, x=X1)
    m = jnp.insert(m, 0, 0) - m[0] #Boundary conditions
    
    y = cumulative_trapezoid(y=m, x=X1)
    y = jnp.insert(y, 0, 0) - y[0] #Boundary conditions
    
    m /= EI
    y /= EI
    
    return (y, m, M, V)