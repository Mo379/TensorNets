#system
import os
from pathlib import Path
#ML
import numpy as np
import jax
import haiku as hk
#env
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
#logging
import wandb
import imageio
#local
from src.saving import load_params
from src.network import *


# Initialising the actor
model= hk.transform(my_actor)
rng = jax.random.PRNGKey(0)
examples = jax.random.normal(rng,(1,84,84,3))
# getting the save parameters
root= Path(__file__).resolve().parent
_path = os.path.join(root,'logs/pkls/params_actor.npz')
params = load_params(_path)
# tracking variable
track =1

#main#
if __name__ == '__main__':
    # setting up tracking run
    if track == 1:
        print('----Tracking----')
        run = wandb.init(
            dir= os.path.join(root,'logs/wandb'),
            project="TensorNets",
            name = "Play_Haiku_trained",
            entity="mo379",
        )
    #setting up the environment
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
        # getting environment step vars
        obs, reward, done, info = env.last()
        obs = obs.reshape((1,) + obs.shape)
        # model forward pass
        act = model.apply(params,rng,obs)[0][0] if not done else None
        act = np.array(act)
        env.step(act)
        # saving image and reward
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
