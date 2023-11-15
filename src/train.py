# train.py
#    aphasia training for jax
# by: Noah Syrkis

# imports
import jax
from jax import jit, vmap, value_and_grad
import optax
from tqdm import tqdm
from functools import partial

from src.model import loss_fn


# functions
grad_fn = jit(value_and_grad(loss_fn))
def update_fn(optimizer, opt_state, params, xb, yb, clip_value=1.0):
    loss, grads = grad_fn(params, xb, yb)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss

def train_fn(rng, opt, opt_state, params, config, train_batches, eval_batches):
    update = jit(partial(update_fn, opt))
    pbar = tqdm(range(config['n_steps']))
    for step in pbar:
        rng, key = jax.random.split(rng)
        xb, yb = next(train_batches)
        opt_state, params, loss = update(opt_state, params, xb, yb)
        pbar.set_description(f'train: {loss:.4f} | eval: {999 if "eval_loss" not in locals() else eval_loss:.4f}')
        if step % (config['n_steps'] // 20) == 0:
            eval_loss = evaluate_fn(train_batches, eval_batches, params)
    return opt_state, params

 
jit_loss_fn = jit(partial(loss_fn))
def evaluate_fn(train_batches, eval_batches, params, n_steps=4):
    train_loss = 0
    eval_loss = 0
    for _ in range(n_steps):
        # xb, yb = next(train_batches)
        # train_loss += jit_loss_fn(xb, yb)
        xb, yb = next(eval_batches)
        eval_loss += jit_loss_fn(params, xb, yb)
    train_loss /= n_steps
    eval_loss /= n_steps
    return eval_loss  # train_loss, eval_loss


    