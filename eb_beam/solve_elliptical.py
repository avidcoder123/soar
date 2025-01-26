import jax
import jax.numpy as jnp

@jax.jit
def shear_force(L, t):
    V = t/2
    V -= 0.5 * jnp.sin(t) * jnp.cos(t)
    V *= 2 * L / jnp.pi
    
    return V

@jax.jit
def moment(L, b, t, C1):
    M = jnp.sin(t)
    M -= t * jnp.cos(t)
    M -= (1/3) * (jnp.sin(t)) ** 3
    M *= L * b / (2 * jnp.pi)

    M -= (b/2) * C1 * jnp.cos(t)
    
    return M
    
@jax.jit
def solve_beam(L, b):
    #Enforce shear at free end = 0
    C1 = -shear_force(L, jnp.pi)
    
    #Enforce moment at free end = 0
    #Always true because of sine properties
    #C2 = -moment(L, b, jnp.pi, C1)
    
    #Get the max shear and moment (at z=0)
    
    # V = shear_force(L, jnp.pi/2) + C1
    # V = jnp.abs(V)
    
    #The shear force at the wing root is always half of that at the wingtip (due to sine properties)
    V = jnp.abs(C1/2)
    M = moment(L, b, jnp.pi/2, C1)# + C2
    M = jnp.abs(M)
    
    return (V, M)