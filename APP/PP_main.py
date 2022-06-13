import os
from src.util import *
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
from pathlib import Path
import wandb
from wandb.integration.sb3 import WandbCallback


#tracking
track =1
#hyper params
verbose=3
gamma=0.95 
ent_coef=0.0905168
learning_rate=0.00062211 
vf_coef=0.042202
max_grad_norm=0.9
gae_lambda=0.99 
clip_range=0.3

n_epochs=5
n_steps=256
batch_size=256
total_timesteps=2000000


n_epochs=10
n_steps=8
batch_size=8
total_timesteps=10000
if __name__ == '__main__':
    if track == 1:
        print('----Tracking----')
        run = wandb.init(
            project="TensorNets",
            name='Training-PPO-SB', 
            entity="mo379",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
    env = environment_setup()
    env = Monitor(env)
    env = VecVideoRecorder(
            env, f"videos/{run.id}", 
            record_video_trigger=lambda x: x % 2000 == 0, video_length=200
        )

    model = PPO(
        CnnPolicy,env,verbose=verbose,gamma=gamma, 
        n_steps=n_steps,ent_coef=ent_coef,learning_rate=learning_rate, 
        vf_coef=vf_coef,max_grad_norm=max_grad_norm,gae_lambda=gae_lambda, 
        n_epochs=n_epochs,clip_range=clip_range,batch_size=batch_size,tensorboard_log=f"runs/{run.id}"
    )
    root= Path(__file__).resolve().parent.parent
    _path = os.path.join(root,'pkls/models/policy')
    if track ==1:
        model.learn(total_timesteps=total_timesteps, callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ))
        run.finish()
    else:
        model.learn(total_timesteps=total_timesteps)
        model.save(_path)
