# system
import os
from pathlib import Path
# ML
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
# env
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
# logging
import wandb
import imageio
# local
from src.agent import my_model_tensornet


max_cycles = 125


def environment_setup():
    # setting up the testing and live cases
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=False,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=max_cycles
    )
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, stack_size=3)
    return env


# Initialising the actor
model = hk.transform(my_model_tensornet)
rng = jax.random.PRNGKey(0)
examples = jax.random.normal(rng, (20, 84, 84, 3))
# getting the save parameters
root = Path(__file__).resolve().parent
params = model.init(rng, examples)
# tracking variable
track = 1

# main
if __name__ == '__main__':
    # setting up tracking run
    if track == 1:
        print('----Tracking----')
        run = wandb.init(
            dir=os.path.join(root, 'logs/wandb'),
            project="TensorNets",
            name="Play_Haiku_tensornet_untrained",
            entity="mo379",
        )
    # setting up the environment
    env = environment_setup()
    obs = env.reset()
    # setting up logging lists
    imgs = []
    rewards = []
    # main playing loop (for all agents)
    dones = jnp.array([0])
    for cycle in range(max_cycles):
        print(f'step: {cycle}')
        obs = [
                jnp.array(observation).astype(jnp.float64)
                for observation in list(obs.values())
            ]
        obs = jnp.array(obs)
        actions = model.apply(params, rng, obs)[0]
        actions = {
                agent: int(actions[idx]) for idx, agent in zip(list(range(len(actions))),env.agents)
            }
        obs, reward, dones, infos = env.step(actions)
        img = env.render(mode='rgb_array')
        imgs.append(img)
        rewards.append(np.mean(list(reward.values())))
    #
    env.reset()
    # logging the run
    if track == 1:
        for reward in rewards:
            # logging reward
            wandb.log({"rewards": reward})
        # making video
        imageio.mimsave(
                os.path.join(root, 'logs/play_videos/0.gif'),
                [np.array(img) for i, img in enumerate(imgs)],
                fps=15
            )
        # saving video
        wandb.log({
                "video": wandb.Video(
                        os.path.join(root, 'logs/play_videos/0.gif'),
                        fps=15,
                        format='gif'
                    )
            })
        run.finish()
