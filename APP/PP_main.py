import os
from src.util import *
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    env = environment_setup()
    model = PPO(
        CnnPolicy,env,verbose=3,gamma=0.95, 
        n_steps=256,ent_coef=0.0905168,learning_rate=0.00062211, 
        vf_coef=0.042202,max_grad_norm=0.9,gae_lambda=0.99, 
        n_epochs=5,clip_range=0.3,batch_size=256
    )
    root= Path(__file__).resolve().parent.parent
    _path = os.path.join(root,'pkls/models/policy')
    model.learn(total_timesteps=2000000)
    model.save(_path)
