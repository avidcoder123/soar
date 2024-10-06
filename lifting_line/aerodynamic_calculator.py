import jax
import jax.numpy as jnp
import jax.scipy as jsci
from fourier_util import circulation_fn, alpha_i_fn

@jax.jit
def calculate_aerodynamics(coefficients, n_list, v_infty, rho, Re, alpha_geo, c, b, B, T, P, C, E, R):
    thetas = jnp.linspace(1e-3, jnp.pi - 1e-3, wing_points)
    z = -(b/2) * jnp.cos(thetas)
    
    gammas = jax.vmap(circulation_fn, in_axes=(0, None, None, None, None))(thetas, coefficients, n_list, b, v_infty).flatten()
    alphas_i = jax.vmap(alpha_i_fn, in_axes=(0, None, None))(thetas, coefficients, n_list)
    alphas_eff = alpha_geo - alphas_i

        
    lift = jsci.integrate.trapezoid(y=gammas, x=z)
    lift *= rho * v_infty
    
    
    induced_drag = jsci.integrate.trapezoid(y=gammas*alphas_i, x=z)
    induced_drag *= rho * v_infty

    #Drag per unit span
    d_primes = get_cds(B, T, P, C, E, R, alphas_eff, Re).flatten()
    d_primes *= (1/2) * rho * v_infty ** 2 * c
    drag = jsci.integrate.trapezoid(y=d_primes, x=z)
    drag += induced_drag
    
    return jnp.array([lift, drag])