import os
from datetime import datetime
import time
from pathlib import Path
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
import wandb
from src.util import *
from src.network import *
from src.ppo import PPO as PPO_jax
from src.env_setup import environment_setup,play_enviromnet_setup
from src.trainer import trainer

#setting up model hyperparams
lr_actor=0.00032211
lr_critic=0.00032211
# setting up algorithm hyperparameters
max_grad_norm = 0.9
gamma=0.95
clip_eps=0.35
lambd=0.97
seed=0
action_repeat=1
#setting training length hyperparams
num_agent_steps=3000000
buffer_size=2048
epochs=5
batch_size=128
# testing scenario
test = True
if test:
    num_agent_steps=1000
    buffer_size=128
    epochs=3
    batch_size=32
# evaluation hyperparams
eval_interval=num_agent_steps//10
num_eval_episodes = 3
save_params=True
#root
root= Path(__file__).resolve().parent


#main
if __name__ == "__main__":
    print('----Tracking----')
    wandb_run= wandb.init(
        dir= os.path.join(root,'logs/wandb'),
        project="TensorNets",
        name = "Train_haiku_ppo_nature",
        entity="mo379",
    )
    # setting up main and test environments
    env = environment_setup(test=test)
    env_eval = environment_setup(test=test)
    env_test = play_enviromnet_setup()
    #algorithm setup
    algo = PPO_jax(
        # models and model hyper params
        fn_actor=my_actor,
        fn_critic=my_critic,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        #algorithm hyper params
        max_grad_norm=max_grad_norm,
        gamma=gamma,
        clip_eps=clip_eps,
        lambd=lambd,
        #env hyperparams
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=seed,
        #training length hyperparams 
        num_agent_steps=num_agent_steps,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epoch_ppo=epochs,
    )
    #setting up run logging directory
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_vars = [os.path.join(root,"logs/Haiku_nature/"), time]
    # setting up the trainer
    trainer = trainer(
        #envs
        env=env,
        env_eval=env_eval,
        env_test=env_test,
        #algorith
        algo=algo,
        #algo hyperparams
        action_repeat=action_repeat,
        num_agent_steps=num_agent_steps,
        seed=seed,
        #logging
        log_vars=log_vars,
        eval_interval=eval_interval,
        num_eval_episodes=num_eval_episodes,
        save_params=save_params,
        wandb_run= wandb_run
    )
    # training
    trainer.train()
    print('done')

    exit()
