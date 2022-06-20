import argparse
import os
from datetime import datetime
import time
from pathlib import Path

from rljax.algorithm import PPO as rljPPO
from rljax.env import make_continuous_env
from rljax.trainer import Trainer

from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np

from src.util import *
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import wandb

def run(args):
    env = environment_setup()
    env_test = environment_setup()

    algo = rljPPO(
        #
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        fn_actor=my_actor,
        fn_critic=my_critic,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        epoch_ppo=args.epochs,
        clip_eps=args.clip_eps,
        lambd=args.lambd,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.env_id, f"{str(algo)}-seed{args.seed}-{time}")

    trainer = my_Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_agent_steps=args.num_agent_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()
    print('done')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="PistonBallv6")
    p.add_argument("--num_agent_steps", type=int, default=1000)#3 * 10 ** 6)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    #
    p.add_argument("--max_grad_norm", type=float, default=0.9)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--buffer_size", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--lr_actor", type=float, default=0.0006)
    p.add_argument("--lr_critic", type=float, default=0.0006)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--clip_eps", type=float, default=0.3)
    p.add_argument("--lambd", type=int, default=0.99)
    args = p.parse_args()
    run(args)
