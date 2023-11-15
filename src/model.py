# model.py
#   aphasia model for jax
# by: Noah Syrkis

# imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from functools import partial
import os
import requests

import numpy as np
import os
from tqdm import tqdm
import time


# functions
def head_fn(params, x):
    x = [head_apply_fn(params['head'][f'head_{i}'], x) for i in range(len(params['head']))]
    x = jnp.concatenate(x, axis=-1)
    x = jnp.dot(x, params['proj'])
    return x

def head_apply_fn(params, x):
    B, T, C = x.shape
    tril = jnp.nan_to_num(jnp.absolute(jnp.tril(jnp.ones((T, T))) - 1) * (-jnp.inf), nan=0)
    # mask = jnp.nan_to_num(jnp.triu(jnp.ones((T, T))) * (-jnp.inf), nan=0)
    H = params['key'].shape[1]
    k = jnp.dot(x, params['key'])       # B x T x H
    q = jnp.dot(x, params['query'])     # B x T x H
    wei = q @ k.transpose(0, 2, 1)      # B x T x T
    wei /= jnp.sqrt(H)                  # normalise
    wei += tril                         # mask future
    wei = jax.nn.softmax(wei, axis=-1)  # B x T x T
    v = jnp.dot(x, params['value'])     # B x T x H
    out = wei @ v                       # B x T x H
    return out

def init_head_fn(rng, n_embed, n_heads, scale=1e-2):
    head_size = n_embed // n_heads
    rng, key_key, key_value, key_query = jax.random.split(rng, 4)
    params = {} 
    for i in range(n_heads):
        params[f'head_{i}'] = {
            'key':   jax.random.normal(key_key,   shape=(n_embed, head_size)) * scale,
            'value': jax.random.normal(key_value, shape=(n_embed, head_size)) * scale,
            'query': jax.random.normal(key_query, shape=(n_embed, head_size)) * scale,
            }
    return params

def ffwd_fn(params, x):
    out = jax.nn.relu(x @ params['dense1'] + params['bias1'])
    out = out @ params['dense2'] + params['bias2']
    return out

def init_ffwd_fn(rng, n_embed, scale=1e-2):
    rng, key1, key2 = jax.random.split(rng, 3)
    params = {
        'dense1': jax.random.normal(key1, shape=(n_embed, 4 * n_embed)) * scale,
        'bias1': jax.random.normal(key1, shape=(4 * n_embed,)) * scale,
        'dense2': jax.random.normal(key2, shape=(4 * n_embed, n_embed)) * scale,
        'bias2': jax.random.normal(key2, shape=(n_embed,)) * scale,
        }
    return params


def layer_norm_fn(params, x, eps=1e-5):
    gamma, beta = params['gamma'], params['beta']
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    out = (x - mean) / (std + eps)
    out = out * gamma + beta
    return out

def init_layer_norm_fn(n_embed):
    params = {
        'gamma': jnp.ones((n_embed,)),
        'beta': jnp.zeros((n_embed,)),
        }
    return params


def init_block_fn(rng, n_embed, n_heads, scale=1e-2):
    rng, key1, key2, key3, key4, key5 = jax.random.split(rng, 6)
    params = {
        'head': init_head_fn(key1, n_embed, n_heads, scale),
        'ffwd': init_ffwd_fn(key2, n_embed, scale),
        'proj': jax.random.normal(key3, shape=(n_embed, n_embed)) * scale,
        'ln1': init_layer_norm_fn(n_embed),
        'ln2': init_layer_norm_fn(n_embed),
        }
    return params

def block_fn(params, x):
    x = layer_norm_fn(params['ln1'], x)
    x += head_fn(params, x)
    x = layer_norm_fn(params['ln2'], x)
    x += ffwd_fn(params['ffwd'], x)
    return x


def apply_fn(params, xb):
    B, T = xb.shape
    tok_embs = params['tok_embedding'][xb]              # B x T x C
    pos_embs = params['pos_embedding'][jnp.arange(T)]   # T x C
    x = tok_embs + pos_embs
    for block in params['blocks']:
        x = block_fn(block, x)
    x = layer_norm_fn(params['layer_norm'], x)
    logits = x @ params['lm_head']                 # B x T x V
    return logits


def init_fn(rng, n_embed, n_heads, vocab_size, block_size, n_layers, scale=1e-2):
    rng, key1, key2, key3, key4, key5 = jax.random.split(rng, 6)
    params = {
        'tok_embedding': jax.random.normal(key1, shape=(vocab_size, n_embed)) * scale,
        'pos_embedding': jax.random.normal(key2, shape=(block_size, n_embed)) * scale,
        'lm_head': jax.random.normal(key3, shape=(n_embed, vocab_size)) * scale,
        'blocks': [init_block_fn(key1, n_embed, n_heads, scale=scale) for _ in range(n_layers)],
        'layer_norm': init_layer_norm_fn(n_embed),
        }

    return params


def loss_fn(params, xb, yb):
    # we cant to minimise cross entropy
    logits = apply_fn(params, xb) # B x T x C
    B, T, C = logits.shape
    yb = yb.reshape(-1)
    logits = logits.reshape(B * T, C)
    logits = jnp.clip(logits, -100, 100)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(yb, C)))
    return loss

    
def generate_fn(rng, params, idx, block_size, length=100, temperature=1.0):
    for _ in tqdm(range(length)):
        rng, key = jax.random.split(rng)
        logits = apply_fn(params, idx[:, -block_size:])         # B x T x C
        logits = logits[:, -1, :] / temperature                 # B x C
        idx_new = jax.random.categorical(key, logits)[:, None]  # B x 1
        idx = jnp.concatenate([idx, idx_new], axis=1)           # B x T + 1
    return idx