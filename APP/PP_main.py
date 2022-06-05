import os
from .src.util import *
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np

env = environment_setup()
model = PPO(
    CnnPolicy, 
    env, 
    verbose=3, 
    gamma=0.95, 
    n_steps=32, 
    ent_coef=0.0905168, 
    learning_rate=0.00062211, 
    vf_coef=0.042202, 
    max_grad_norm=0.9, 
    gae_lambda=0.99, 
    n_epochs=2, 
    clip_range=0.3, 
    batch_size=32
)
model.learn(total_timesteps=20000)
_path = os.path.join(root,'Tutorials/models/policy')
model.save(_path)
env = pistonball_v6.env()
env = ss.color_reduction_v0(env,mode='B')
env = ss.resize_v1(env, x_size=84,y_size=84)
env = ss.frame_stack_v1(env, 4)
model = PPO.load(_path)
if __name__ == '__main__':
    print('hello world')
