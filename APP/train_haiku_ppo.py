#system 
import os
import time
from pathlib import Path
#ML
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
#log
import wandb
#local
from src.agent import *
from src.ppo import PPO
from src.util import trainer,environment_setup,play_enviromnet_setup

#seed and root
seed=0
root= Path(__file__).resolve().parent
#loss scales for entropy and value losses
ent_coef=0.0905168
vf_coef=0.042202
#setting up model hyperparams
lr_policy=0.0001
# setting up algorithm hyperparameters
max_grad_norm = 0.9
gamma=0.95
clip_eps=0.2
lambd=0.99
#setting training length hyperparams
num_agent_steps=10000
buffer_size=64
epochs=10
batch_size=256
# testing scenario
test = True
if test:
    num_agent_steps=5000
    buffer_size=32
    epochs=5
    batch_size=256
# evaluation hyperparams
eval_interval=num_agent_steps//10
num_eval_episodes = 5
save_params=True
#
track = True


#main
if __name__ == "__main__":
    if track:
        print('----Tracking----')
        wandb_run= wandb.init(
            dir= os.path.join(root,'logs/wandb'),
            project="TensorNets",
            name = "Train_haiku_ppo_nature",
            entity="mo379",
        )
    else:
        wandb_run = False
    # setting up main and test environments
    env = environment_setup(test=test)
    env_eval = environment_setup(test=test)
    env_test = play_enviromnet_setup()
    #algorithm setup
    algo = PPO(
        #seed and root
        seed=seed,
        root=root,
        # models and model hyper params
        fn_policy=my_model,
        lr_policy=lr_policy,
        #algorithm hyper params
        max_grad_norm=max_grad_norm,
        gamma=gamma,
        clip_eps=clip_eps,
        lambd=lambd,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        #env hyperparams
        state_space=env.observation_space,
        action_space=env.action_space,
        #training length hyperparams 
        num_agent_steps=num_agent_steps,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epoch_ppo=epochs,
    )
    # setting up the trainer
    trainer = trainer(
        #seed and root
        seed=seed,
        root=root,
        #envs
        env=env,
        env_eval=env_eval,
        env_test=env_test,
        #algorithm
        algo=algo,
        #algo hyperparams
        num_agent_steps=num_agent_steps,
        #logging
        eval_interval=eval_interval,
        num_eval_episodes=num_eval_episodes,
        save_params=save_params,
        wandb_run= wandb_run
    )
    # training
    trainer.train()
    print('done')

    exit()
