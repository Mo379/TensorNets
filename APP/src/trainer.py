import os
from datetime import timedelta
from time import time

import os
import argparse
from datetime import datetime
from pathlib import Path
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
import wandb
from .util import *
from .ppo import PPO as PPO_jax
import imageio



class trainer:
    """
    Trainer.
    """
    def __init__(
        self,
        #envs
        env,
        env_eval,
        env_test,
        #algo
        algo,
        # algo hyper
        action_repeat,
        num_agent_steps,
        seed,
        #loggin
        log_vars,
        eval_interval,
        num_eval_episodes,
        save_params,
        wandb_run,
    ):
        #making sure some key variables are balanced
        assert num_agent_steps % action_repeat == 0
        assert eval_interval % action_repeat == 0

        # Envs.
        self.env = env
        self.env_eval = env_eval
        self.env_test = env_test

        # Set seeds.
        self.env.seed(seed)
        self.env.seed(seed+1)
        self.env_test.reset(seed=seed+2)

        # Algorithm.
        self.algo = algo

        # Log setting.
        self.log_dir = log_vars[0]
        self.log_id = log_vars[1]
        # logs
        self.param_dir = os.path.join(self.log_dir, "param", self.log_id)
        self.wandb_run = wandb_run

        # Other parameters.
        self.action_repeat = action_repeat
        self.num_agent_steps = num_agent_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.save_params = save_params

    #training function
    def train(self):
        # Time to start training.
        self.start_time = time()
        # Initialize the environment.
        state = self.env.reset()
        #start counting steps
        done = np.array([0])
        for step in range(1, self.num_agent_steps + 1):
            #take a step and load the buffer
            while self.algo.is_update(state.shape[0]) == True:
                print(f"step {step}")
                state,done = self.algo.step(self.env, state,done)
            print('learning')
            self.algo.update(self.wandb_run)
            #if we are at a step where evaluation is wanted then evaluate
            print(f"step-{step}, interval {self.eval_interval}: evaluation modulus {step % self.eval_interval}")
            if step % self.eval_interval == 1:
                #evaluate current model
                self.evaluate(step)
                # save current model and log
                if self.save_params:
                    print('saving_params')
                    params_path = os.path.join(self.param_dir, f"step{step}")
                    #save to params dir
                    self.algo.save_params(params_path)
                    #log params dir
                    artifact = wandb.Artifact('params', type='params')
                    #add to wandb
                    artifact.add_dir(params_path)
                    self.wandb_run.log_artifact(artifact)
                    print('loging model params as histograms')
                    params = self.algo.params_actor.copy()
                    log_std = params['log_std']['constant']
                    params.pop('log_std')
                    wandb.log({"params-log_std":wandb.Histogram(log_std)})
                    for layer in params:
                        w = params[layer]['w']
                        b = params[layer]['b']
                        wandb.log({f"params-{layer}-weights":wandb.Histogram(w)})
                        wandb.log({f"params-{layer}-bias":wandb.Histogram(b)})

    #evaluation function
    def evaluate(self, step):
        total_return = 0.0
        # run n-episodes
        for i_counter in range(self.num_eval_episodes):
            print(f"Eval episode: {i_counter}")
            #reset state
            state = self.env_eval.reset()
            done = np.array([0])
            #run until done
            while done.any() == False: 
                #get action
                action = self.algo.select_action(state)
                #get environemnt observables
                state, reward, done, _ = self.env_eval.step(action) if not done.any() else None
                total_return += np.mean(reward)
        # Log mean return and step.
        mean_return = total_return / self.num_eval_episodes
        wandb.log({"step": step * self.action_repeat})
        wandb.log({"mean_return": mean_return})


        self.env_test.reset()
        #setting up logging lists
        imgs = []
        rewards= []
        # main playing loop (per agent)
        for agent in self.env_test.agent_iter():
            # getting environment step vars
            obs, reward, done, info = self.env_test.last()
            # model forward pass
            act = self.algo.select_action(obs)[0] if not done else None
            act = np.array(act)
            self.env_test.step(act)
            # saving image and reward
            img = self.env_test.render(mode='rgb_array')
            imgs.append(img)
        #
        #
        imageio.mimsave(
                os.path.join(self.log_dir,f"videos/{self.log_id}.gif"), 
                [np.array(img) for i, img in enumerate(imgs) if i%10 == 0], 
                fps=15
        )
        #saving video
        wandb.log({
            "videos": wandb.Video(os.path.join(self.log_dir,f"videos/{self.log_id}.gif"),
            fps=15,
            format='gif')}
        )


    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))








