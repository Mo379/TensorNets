import os
from src.util import *
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
from pathlib import Path

#hyper params
verbose=3
gamma=0.95 
n_steps=256
ent_coef=0.0905168
learning_rate=0.00062211 
vf_coef=0.042202
max_grad_norm=0.9
gae_lambda=0.99 
n_epochs=5
clip_range=0.3
batch_size=256
#
track =0
if track == 1:
    wandb.init(project="TensorNets", entity="mo379")
    wandb.config = {
        "name": 'train-PPO',
        "verbose":verbose
        "gamma":gamma
        "n_steps":n_steps
        "ent_coef":ent_coef
        "learning_rate":learning_rate
        "vf_coef":vf_coef
        "max_grad_nrom":max_grad_norm
        "gae_lambda":gae_lambda
        "n_epochs":n_epochs
        "clip_range":clip_range
        "batch_size":batch_size
    }
if __name__ == '__main__':
    pass
