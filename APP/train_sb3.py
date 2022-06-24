import os
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

from src.env_setup import *
from time import sleep


#model hyperparams
learning_rate=0.00032211
#algorithm hyper params
verbose=3
gamma=0.95
ent_coef=0.0905168
vf_coef=0.042202
max_grad_norm=0.9
gae_lambda=0.99 
clip_range=0.3
# training length hyperparams
n_epochs=5
n_steps=256
batch_size=256
total_timesteps=2000000
#testing case hyperparameters
test = True
if test:
    n_epochs=5
    n_steps=256
    batch_size=256
    total_timesteps=2000
eval_freq = total_timesteps//10


#root
root= Path(__file__).resolve().parent




#main#
if __name__ == '__main__':
    print('----Tracking----')
    run = wandb.init(
        dir= os.path.join(root,'logs/wandb'),
        project="TensorNets",
        name = "Training-SB3-PPO",
        entity="mo379",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    # setting up the environment with logging
    env = environment_setup(test=test)
    env = Monitor(env)
    env = VecVideoRecorder(
            env, os.path.join(root,f"logs/SB3/videos/{run.id}"), 
            record_video_trigger=lambda x: x % eval_freq == 0, video_length=250
        )
    #algorithm setup
    model = PPO(
        CnnPolicy,env,verbose=verbose,gamma=gamma, 
        n_steps=n_steps,ent_coef=ent_coef,learning_rate=learning_rate, 
        vf_coef=vf_coef,max_grad_norm=max_grad_norm,gae_lambda=gae_lambda, 
        n_epochs=n_epochs,clip_range=clip_range,batch_size=batch_size,
        tensorboard_log=os.path.join(root,f"logs/SB3/tensorboard/{run.id}")
    )
    # learning and tracking
    model.learn(total_timesteps=total_timesteps, callback=WandbCallback(
            gradient_save_freq=eval_freq,
            model_save_freq=eval_freq,
            model_save_path=os.path.join(root,f"logs/SB3/models/policy_{run.id}"),
            verbose=2,
        ))
    run.finish()
    exit()
