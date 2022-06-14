import os
import time
from pathlib import Path
import wandb
import numpy as np
import jax
import jax.numpy as jnp 
import haiku as hk
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from pettingzoo.utils import random_demo
import imageio
from src.util import *

root= Path(__file__).resolve().parent.parent
_path = os.path.join(root,'pkls/models/policy')
model = PPO.load(_path)

track =1


if __name__ == '__main__':
    if track == 1:
        print('----Tracking----')
        run = wandb.init(
            project="TensorNets",
            name = "Play_PPO",
            entity="mo379",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
    env = pistonball_v6.env()
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env.reset()
    imgs = []
    rewards = []
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        img = env.render(mode='rgb_array')
        imgs.append(img)
        rewards.append(reward)
    env.reset()
    #
    for reward in rewards:
        wandb.log({"rewards": reward})
    imageio.mimsave('play_videos/0.gif', [np.array(img) for i, img in enumerate(imgs) if i%30 == 0], fps=30)
    wandb.log({"video": wandb.Video('play_videos/0.gif',fps=30,format='gif')})
    run.finish()






