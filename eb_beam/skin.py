import jax
import jax.numpy as jnp

#t is the skin thickness, in c
#T is the max thickness of the airfoil, in c
#c is the chord length in meters
#alpha is angle of attack in degrees
@jax.jit
def skin_moments(t, T, c, alpha):
    alpha = jnp.deg2rad(alpha)
    max_y = 0.25 * jnp.sin(alpha) ** 2
    max_y += (T/2) ** 2 * jnp.cos(alpha) ** 2
    max_y = jnp.sqrt(max_y)
    
    A = jnp.pi * (T/2) * (1/2) #Cross-section area
    A -= jnp.pi * ((T-t)/2) * ((1-t)/2)
    
    T = 2 * max_y #Adjust the dimensions of the airfoil ellipse based on rotation
    
    I = (1/6) * t * T ** 3
    Q0 = t * T ** 2 / 4
    
    #Unit conversion
    I *= c ** 4
    Q0 *= c ** 3
    A *= c ** 2
    
    return (I, Q0, A)