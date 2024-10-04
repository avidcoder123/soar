import equinox as eqx
import jax
import jax.numpy as jnp

input_labels = ["B", "T", "P", "C", "E", "R", "Alpha", "Re"]

#Define the structure for the neural network
class SurrogateModel(eqx.Module):
    
    layers: list
        
    def __init__(self, in_size, out_size, width_size, depth, activation, key):
        keys = jax.random.split(key, depth + 2)
        
        input_key = keys[0]
        output_key = keys[-1]
        hidden_keys = keys[1:-1]
        
        input_layer = eqx.nn.Linear(in_size, width_size, key=input_key)
        output_layer = eqx.nn.Linear(width_size, out_size, key=output_key)
        
        #Make Reynolds number on log10 scale
        @jax.jit
        def normalize_reynolds_number(x):
            Re = x[-1]
            Re = jnp.log10(Re)
            
            #Set maximum of Re=10^6
            Re = jnp.min(jnp.hstack((Re, 6)))
            
            return jnp.hstack((x[:-1], Re))
            
        
        self.layers = [
            normalize_reynolds_number,
            jax.nn.standardize, #Standardize -1 to 1
            input_layer,
            activation
        ]
        for key in hidden_keys:
            self.layers.append(eqx.nn.Linear(width_size, width_size, key=key))
            self.layers.append(activation)
            
        self.layers.append(output_layer)
        
    def __call__(self, x):
                
        for layer in self.layers:
            x = layer(x)

        return x