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



env = pistonball_v6.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env.reset()

if __name__ == '__main__':
    random_demo(env, render=True, episodes=1)
