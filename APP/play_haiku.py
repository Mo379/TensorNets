import os
import time
from pathlib import Path
import pickle
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
from rljax.util import load_params
from src.util import *


# Initialising the feature extractor
model_features= hk.transform(my_model)
rng = jax.random.PRNGKey(0)
examples = jax.random.normal(rng,(1,84,84,3))
model_features_params = model_features.init(rng, examples)
# Transfer parameters from the saved model
root= Path(__file__).resolve().parent.parent
_path = os.path.join(root,'pkls/models/model')
j_path = os.path.join(root,'pkls/models/params_actor.npz')
k_path = os.path.join(root,'pkls/models/params_actor2.npz')
model = PPO.load(_path)
trained_params = model.get_parameters()
transferred_params = transfer_params(trained_params['policy'], model_features_params)


_path_pkl = os.path.join(root,'pkls/models/haiku_transfer.pkl')
# Saving the objects:
with open(_path_pkl, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(transferred_params, f)

# making haiku model
model= hk.transform(my_actor)
rng = jax.random.PRNGKey(0)
examples = jax.random.normal(rng,(1,84,84,3))
#Loading haiku params
root= Path(__file__).resolve().parent.parent
file = open(_path_pkl, 'rb')
#params = pickle.load(file)
params = load_params(j_path)
file.close()
#


track =1


if __name__ == '__main__':
    if track == 1:
        print('----Tracking----')
        run = wandb.init(
            project="TensorNets",
            name = "Play_Haiku",
            entity="mo379",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
    env = pistonball_v6.env(n_pistons=20)
    env = ss.color_reduction_v0(env,mode='B')
    env = ss.resize_v1(env, x_size=84,y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env.reset()

    imgs = []
    rewards= []
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        obs = obs.reshape((1,) + obs.shape)
        act = model.apply(params,rng,obs)[0][0] if not done else None
        act = np.array(act)
        env.step(act)
        img = env.render(mode='rgb_array')
        imgs.append(img)
        rewards.append(reward)
    env.reset()
    #
    if track == 1:
        for reward in rewards:
            wandb.log({"rewards": reward})
        imageio.mimsave('play_videos/0.gif', [np.array(img) for i, img in enumerate(imgs) if i%20 == 0], fps=15)
        wandb.log({"video": wandb.Video('play_videos/0.gif',fps=15,format='gif')})

        run.finish()
