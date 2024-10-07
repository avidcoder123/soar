import jax
import jax.numpy as jnp
from functools import partial
from surrogate import SurrogateModel
import equinox as eqx

input_labels = ["B", "T", "P", "C", "E", "R", "Alpha", "Re"]

#Generate the base network, later to be populated with weights
def generate_base_model(key=jax.random.PRNGKey(42)):
    model = SurrogateModel(
        in_size=len(input_labels),
        out_size=1,
        width_size=64,
        depth=4,
        activation=jax.nn.silu,
        key=key
    )
    
    return model

#These utility functions take multiple arguments as input rather than an array.
#They also take angle inputs in radians and convert it to degrees.
#This is because the neural network is trained in degrees.

@eqx.filter_jit
def cl(model, B, T, P, C, E, R, alpha, Re):
    alpha = jnp.rad2deg(alpha)
    return model(jnp.hstack([B, T, P, C, E, R, alpha, Re]))

@eqx.filter_jit
def cd(model, B, T, P, C, E, R, alpha, Re):
    alpha = jnp.rad2deg(alpha)
    return model(jnp.hstack([B, T, P, C, E, R, alpha, Re]))

#Get Cd but vectorize wrt theta
get_cds = jax.vmap(cd, in_axes=(None, None, None, None, None, None, None, 0, None))