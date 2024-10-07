import jax
import jax.numpy as jnp
import optax
from .fourier_util import alpha_i_fn
from functools import partial

#Set up optimization for circulation optimization
@jax.jit
def circulation_error(theta, coefficients, n_list, alpha_0, alpha_geo, b, c):
    alpha_eff = alpha_0

    summation_fn = jax.vmap(lambda An, n: An * jnp.sin(n * theta))

    alpha_eff += (2*b)/(jnp.pi * c) * summation_fn(coefficients, n_list).sum()

    #summation_fn = jax.vmap(lambda An, n: n * An * jnp.sin(n * theta) / jnp.sin(theta))
    alpha_i = alpha_i_fn(theta, coefficients, n_list)

    error = jnp.rad2deg(alpha_eff + alpha_i - alpha_geo)

    return error
    
@partial(jax.jit, static_argnums=(2))
def circulation_loss(coefficients, n_list, wing_points, alpha_0, alpha_geo, b, c):
    thetas = jnp.linspace(1e-3, jnp.pi - 1e-3, wing_points)
    error = jax.vmap(circulation_error, in_axes=(0, None, None, None, None, None, None))

    loss = jnp.mean(error(thetas, coefficients, n_list, alpha_0, alpha_geo, b, c) ** 2) #MSE loss
    
    return loss

def solve_coefficients(n_list, wing_points, v_infty, rho, mu, alpha_geo, c, b, alpha_0):
    
    #Initial guess for Fourier coefficients
    coefficients = jnp.array([1.6569553e-02, 2.6246815e-03, 6.8349997e-04, 2.1998581e-04, 8.2535254e-05, 3.4354089e-05, 1.4852858e-05, 6.4759911e-06, 4.3458158e-06, 2.8336714e-05])
    
    solver = optax.adabelief(1e-3)
    opt_state = solver.init(coefficients)
    
    value_and_grad = jax.value_and_grad(circulation_loss)
    cost = 1e10

    for i in range(100):
        cost, grad = value_and_grad(coefficients, n_list, wing_points, alpha_0, alpha_geo, b, c)

        updates, opt_state = solver.update(
            grad,
            opt_state
        )

        coefficients = optax.apply_updates(coefficients, updates)
            
    return coefficients