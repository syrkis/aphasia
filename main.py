# main.py
#   aphasia main file for jax
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax

import numpy as np

from src.model import init_fn, apply_fn
from src.data import get_batches
from src.utils import config


# functions
def main():
    rng, key = jax.random.split(jax.random.PRNGKey(0))
    batches = get_batches(key, 'train')
    params = init_fn(rng, config)
    


if __name__ == '__main__':
    main()
