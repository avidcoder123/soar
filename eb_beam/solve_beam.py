import jax
import jax.numpy as jnp
from scipy.integrate import cumulative_trapezoid, trapezoid

#q is the load distribution in N/m at several points along the wing
#EI is the flexural rigidity (Young's Modulus * 2nd Moment of Area)
#X1 is the length of the beam (length of a single wing)
def solve_beam(q, EI, X1):
    V = jnp.array(cumulative_trapezoid(y=q, x=X1))
    V = jnp.insert(V, 0, 0) - V[-1] #Boundary conditions
    
    M = jnp.array(cumulative_trapezoid(y=V, x=X1))
    M = jnp.insert(M, 0, 0) - M[-1] #Boundary conditions
    
    m = jnp.array(cumulative_trapezoid(y=M, x=X1))
    m = jnp.insert(m, 0, 0) - m[0] #Boundary conditions
    
    y = jnp.array(cumulative_trapezoid(y=m, x=X1))
    y = jnp.insert(y, 0, 0) - y[0] #Boundary conditions
    
    m /= EI
    y /= EI
    
    return (y, m, M, V)