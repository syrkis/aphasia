# data.py
#    conrad text data for jax
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from functools import partial
import os
import yaml


# constants
with open('conrad.txt', 'r') as f:
    text = f.read()

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

chars = sorted(list(set(text)))
vocab_size = len(chars)

c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}
encode = lambda x: [c2i[c] for c in x]
decode = lambda x: ''.join([i2c[i] for i in x])

# data = jnp.array(encode(text))
data = jnp.load('data.npy')
n = int(len(data) * 0.8)
train_data = data[:n]
val_data = data[n:]

# functions
def get_batches(rng, split, buffer=128):
    data = train_data if split == 'train' else val_data
    while True:
        rng, key = jax.random.split(rng)
        idxs = jax.random.randint(key, (buffer, config['batch_size']), 0, len(data) - config['block_size'])
        for i in range(buffer):
            idx = idxs[i][:, None] + jnp.arange(config['block_size'])
            batch = data[idx]
            xb, yb = batch[:, :-1], batch[:, 1:]
            yield xb, yb