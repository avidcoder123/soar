import jax
import jax.numpy as jnp

#V is shear distribution
#M is moment distribution
#T is max airfoil thickness in c
#t is skin thickness in c
#c is chord length
#main_web_w and rear_web_w are web widths of main and rear spars, in c
#I is 2nd moment of area
#Q is 1st moment of area at neutral surface
@jax.jit
def max_stress(V, M, T, t, c, main_web_w, rear_web_w, I, Q0):
    #Width of the total material at the neutral surface
    neutral_t = t * 2 #Wing skin
    neutral_t = main_web_w + rear_web_w #Spars

    neutral_t *= c
    T *= c
    
    normal_stress = jnp.abs(M) * (T / 2) / I
    shear_stress = jnp.abs(V) * Q0 / (I * neutral_t)
    
    max_normal = jnp.max(normal_stress)
    max_shear = jnp.max(shear_stress)
    
    return (max_normal, max_shear)