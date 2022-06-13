import os
import time
from src.util import *
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
import os 
from pettingzoo.utils import random_demo
from pathlib import Path

root= Path(__file__).resolve().parent.parent
_path = os.path.join(root,'pkls/models/policy')
model = PPO.load(_path)


env = pistonball_v6.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env.reset()

track =0
if track == 1:
    wandb.init(project="TensorNets", entity="mo379")
    wandb.config = {
        "name": 'play-PPO',
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 128
    }

if __name__ == '__main__':
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        env.render()
        if track == 1:
            wandb.log({"loss": loss})
