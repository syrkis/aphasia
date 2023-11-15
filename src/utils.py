# utils.py
#    aphasia utils for jax
# by: Noah Syrkis

# imports
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)