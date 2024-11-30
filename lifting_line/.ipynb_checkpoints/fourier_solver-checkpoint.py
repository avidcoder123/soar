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

def solve_coefficients(n_list, wing_points, alpha_geo, c, b, alpha_0):
        
    #Initial guess for Fourier coefficients
    coefficients = jnp.array([1.6569553e-02, 2.6246815e-03, 6.8349997e-04, 2.1998581e-04, 8.2535254e-05, 3.4354089e-05, 1.4852858e-05, 6.4759911e-06, 4.3458158e-06, 2.8336714e-05])
    # coefficients = jnp.array([2.8647788e-02, 4.5476495e-03, 1.1916839e-03, 3.9002407e-04,
    #    1.5229803e-04, 6.8792993e-05, 3.4899742e-05, 1.9403215e-05,
    #    1.1577960e-05, 7.3170331e-06, 4.8295083e-06, 3.3135157e-06,
    #    2.3345781e-06, 1.6940496e-06, 1.2487643e-06, 9.4800913e-07,
    #    7.3570760e-07, 6.2891706e-07, 8.0011984e-07, 6.6432403e-06])
    
    solver = optax.adagrad(1e-2)
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