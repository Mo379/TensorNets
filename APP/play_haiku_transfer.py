import os
import time
from pathlib import Path
import pickle
import wandb
import numpy as np
import jax
import haiku as hk
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
import imageio
from src.util import *
from src.network import *


# Initialising the full model
model_features= hk.transform(my_model)
rng = jax.random.PRNGKey(0)
examples = jax.random.normal(rng,(1,84,84,3))
model_features_params = model_features.init(rng, examples)
# Transfer parameters from the saved model
root= Path(__file__).resolve().parent
_path = os.path.join(root,'logs/pkls/model')
model = PPO.load(_path)
trained_params = model.get_parameters()
transferred_params = transfer_params(trained_params['policy'], model_features_params)
#save transferred haiku params 
_path_pkl = os.path.join(root,'logs/pkls/haiku_transfer.pkl')
with open(_path_pkl, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(transferred_params, f)
# making haiku model
model= hk.transform(my_actor)
rng = jax.random.PRNGKey(0)
examples = jax.random.normal(rng,(1,84,84,3))
#Loading haiku params to haiku model
file = open(_path_pkl, 'rb')
params = pickle.load(file)
file.close()
#tracking variable
track =1
#main#
if __name__ == '__main__':
    # setting up tracking run
    if track == 1:
        print('----Tracking----')
        run = wandb.init(
            dir= os.path.join(root,'logs/wandb'),
            project="TensorNets",
            name = "Play_Haiku_transfer",
            entity="mo379",
        )
    #setting up environment
    env = pistonball_v6.env(n_pistons=20)
    env = ss.color_reduction_v0(env,mode='B')
    env = ss.resize_v1(env, x_size=84,y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env.reset()
    #setting up logging lists
    imgs = []
    rewards= []
    # main playing loop (per agent)
    for agent in env.agent_iter():
        # getting environemnt step vars
        obs, reward, done, info = env.last()
        obs = obs.reshape((1,) + obs.shape)
        # model forward
        act = model.apply(params,rng,obs)[0][0] if not done else None
        act = np.array(act)
        env.step(act)
        #saving image and reward
        img = env.render(mode='rgb_array')
        imgs.append(img)
        rewards.append(reward)
    #
    env.reset()
    #logging the run
    if track == 1:
        for reward in rewards:
            #logging reward
            wandb.log({"rewards": reward})
        #making video
        imageio.mimsave(
                os.path.join(root,'logs/play_videos/0.gif'), 
                [np.array(img) for i, img in enumerate(imgs) if i%20 == 0], 
                fps=15
        )
        #saving video
        wandb.log({
            "video": wandb.Video(os.path.join(root,'logs/play_videos/0.gif'),
            fps=15,
            format='gif')}
        )
        run.finish()
