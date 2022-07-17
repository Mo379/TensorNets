# system
import os
from datetime import timedelta, datetime
from time import time
# ML
import numpy as np
# Env
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
# Log
import wandb
import imageio


# Training environment setup
def environment_setup():
    # setting up the testing and live cases
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125
    )
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, stack_size=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    return env


# Play environment setup
def play_enviromnet_setup():
    env = pistonball_v6.env(n_pistons=20)
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    return env


# Rollout buffer
class RolloutBuffer:
    """
    Rollout Buffer.
    """

    # Init function
    def __init__(
        self,
    ):
        self.state = []
        self.action = []
        self.log_prob = []
        self.reward = []
        self.done = []
        self.next_state = []

    # Appending function
    def append(
            self,
            state,
            action,
            log_prob,
            reward,
            done,
            next_state
    ):
        self.state.append(state)
        self.action.append(action)
        self.log_prob.append(log_prob)
        self.reward.append(reward)
        self.done.append(done)
        self.next_state.append(next_state)

    # Get function
    def get(self):
        return [
            self.state,
            self.action,
            self.log_prob,
            self.reward,
            self.done,
            self.next_state,
        ]

    # Clear function
    def clear(self):
        self.state = []
        self.action = []
        self.log_prob = []
        self.reward = []
        self.done = []
        self.next_state = []


# Trainer
class trainer:
    """
    Trainer.
    """
    # init function
    def __init__(
        self,
        # seed and root
        seed,
        root,
        # envs
        env,
        env_eval,
        env_test,
        # algo
        algo,
        # algo hyper
        num_agent_steps,
        # loggin
        eval_interval,
        num_eval_episodes,
        save_params,
        wandb_run,
    ):
        # Seed and Root.
        self.seed = seed
        self.root = root
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
        self.log_dir = os.path.join(root, "logs/Haiku_nature/")
        self.log_id = datetime.now().strftime("%Y%m%d-%H%M")
        # logs
        self.param_dir = os.path.join(self.log_dir, "param", self.log_id)
        self.wandb_run = wandb_run

        # Other parameters.
        self.num_agent_steps = num_agent_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.save_params = save_params

    # training function
    def train(self):
        # Time to start training.
        self.start_time = time()
        # Initialize the environment.
        state = self.env.reset()
        done = np.array([0])
        #
        for step in range(1, self.num_agent_steps + 1):
            # verbose
            print(
                    f"step-{step}, interval {self.eval_interval}: \
                    evaluation modulus {step % self.eval_interval}"
                )
            # take a step and load the buffer
            state, done = self.algo.step(self.env, state, done)
            #
            if step % self.algo.buffer_size == 0:
                print('learning')
                for output in self.algo.buffer.get():
                    output = np.array(output)
                self.algo.update(self.wandb_run)
                self.algo.buffer.clear()
            # if we are at a step where evaluation is wanted then evaluate
            if step % self.eval_interval == 0:
                # evaluate current model
                self.evaluate(step)
                # save current model and log
                self.save_params_logging(step)

    # evaluation function
    def evaluate(self, step):
        # play with exploration
        self._explorative_play_logging(step)
        # play without exploration and record
        imgs = self._explorative_play()
        # save video
        self._save_videos(imgs)

    #
    def save_params_logging(self, step):
        if self.save_params:
            print('saving_params')
            params_path = os.path.join(self.param_dir, f"step{step}")
            # save to params dir
            self.algo.fn_save_params(params_path)
            if self.wandb_run:
                # log params dir
                artifact = wandb.Artifact('params', type='params')
                # add to wandb
                artifact.add_dir(params_path)
                self.wandb_run.log_artifact(artifact)
                print('loging model params as histograms')
                params = self.algo.params_policy.copy()
                log_std = params['log_std']['constant']
                params.pop('log_std')
                wandb.log({
                        "params-log_std": wandb.Histogram(log_std)
                    })
                for layer in params:
                    w = params[layer]['w']
                    b = params[layer]['b']
                    wandb.log({f"params-{layer}-weights": wandb.Histogram(w)})
                    wandb.log({f"params-{layer}-bias": wandb.Histogram(b)})

    # Explorative play with wandb logging
    def _explorative_play_logging(self, step):
        total_return = 0.0
        # run n-episodes
        for i_counter in range(self.num_eval_episodes):
            print(f"Eval episode: {i_counter}")
            # reset state
            state = self.env_eval.reset()
            done = np.array([0])
            # run until done
            while done.all() is False:
                # get action
                action, log_prob = self.algo.explore(state)
                # get environemnt observables
                state, reward, done, _ = \
                    self.env_eval.step(action) if not done.all() else None
                total_return += reward[0]
        # Log mean return and step.
        mean_return = total_return / self.num_eval_episodes
        if self.wandb_run:
            wandb.log({"step": step})
            wandb.log({"mean_return": mean_return})

    # deterministic explorative play 
    def _non_explorative_play(self):
        self.env_test.reset()
        # setting up logging lists
        imgs = []
        # main playing loop (per agent)
        for agent in self.env_test.agent_iter():
            obs, reward, done, info = self.env_test.last()
            obs = obs.reshape((1,) + obs.shape)
            act = self.algo.select_action(obs)[0][0] if not done else None
            act = np.array(act)
            self.env_test.step(act)
            img = self.env_test.render(mode='rgb_array')
            imgs.append(img)
        return imgs

    # Non-deterministic play
    def _explorative_play(self):
        self.env_test.reset()
        # setting up logging lists
        imgs = []
        # reset state
        state = self.env_eval.reset()
        done = np.array([0])
        # run until done
        while done.all() is False:
            # get action
            action, log_prob = self.algo.explore(state)
            # get environemnt observables
            state, reward, done, _ = \
                self.env_eval.step(action) if not done.all() else None
            img = self.env_eval.render(mode='rgb_array')
            imgs.append(img)
        return imgs

    # Make and Save video function given an array of images
    def _save_videos(self, imgs):
        imageio.mimsave(
                os.path.join(self.log_dir, f"videos/{self.log_id}.gif"),
                [np.array(img) for i, img in enumerate(imgs) if i % 1 == 0],
                fps=15
        )
        if self.wandb_run:
            # saving video
            wandb.log({
                "videos": wandb.Video(
                    os.path.join(self.log_dir, f"videos/{self.log_id}.gif"),
                    fps=15,
                    format='gif'
                )
            })

    # Time property
    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
