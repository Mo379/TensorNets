import os
from pathlib import Path
import wandb
import numpy as np
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
import imageio

#getting the model
root= Path(__file__).resolve().parent
_path = os.path.join(root,'logs/pkls/model')
model = PPO.load(_path)
#setting up the tracking variable
track =1

#main#
if __name__ == '__main__':
    #setting up the tracking run
    if track == 1:
        print('----Tracking----')
        run = wandb.init(
            dir= os.path.join(root,'logs/wandb'),
            project="TensorNets",
            name = "Play_sb3_trained",
            entity="mo379",
        )
    #setting up the environment
    env = pistonball_v6.env(n_pistons=20)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env.reset()
    #setting up logging lists
    imgs = []
    rewards = []
    # main playing loop (per agent)
    for agent in env.agent_iter():
        # getting environemnt step variables
        obs, reward, done, info = env.last()
        #model forward
        act = model.predict(obs, deterministic=True)[0] if not done else None
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






