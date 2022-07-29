#%%
import numpy as np
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss

from jax.example_libraries import stax
from jax.example_libraries.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax)
from jax import random
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from src.tensornet import policy_head, value_function_head

#%%
# env = pistonball_v6.env(n_pistons=20)
env = pistonball_v6.parallel_env(n_pistons=20, time_penalty=-0.1, continuous=False, random_drop=True, random_rotate=True,
 ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v1(env, x_size=84, y_size=84)
observations = env.reset()
#%%
num_classes = 64
batch_size=1

key = random.PRNGKey(0)
init_fun, conv_net = stax.serial(Conv(32, (5, 5), (2, 2), padding="SAME"),
                                 BatchNorm(), Relu,
                                 Conv(32, (5, 5), (2, 2), padding="SAME"),
                                 BatchNorm(), Relu,
                                 Conv(10, (3, 3), (2, 2), padding="SAME"),
                                 BatchNorm(), Relu,
                                 Conv(10, (3, 3), (2, 2), padding="SAME"), Relu,
                                 Flatten,
                                 Dense(num_classes),
                                 LogSoftmax)
_, params = init_fun(key, (batch_size, 1, observations['piston_0'].shape[0], observations['piston_0'].shape[1]))

 # %%
obs = [jnp.array(observation).astype(jnp.float64) for observation in list(observations.values())]
obs = jnp.array(obs)
obs = jnp.reshape(obs, (20,1,84,84))
# %%
plt.imshow(observations['piston_0'])
#%%
plt.imshow(obs[0,0,:,:])
#%%
cnn_output = conv_net(params, obs)
#%%
plt.imshow(cnn_output[:,:])
#%%
key, subkey = random.split(key)
policy_weights = random.normal(subkey, (20, 64, 3, 16, 16))
key, subkey = random.split(key)
vf_weights = random.normal(subkey, (20,64,4,4))
values = value_function_head(vf_weights, cnn_output)
log_prob, (action, key) = policy_head(cnn_output, key, policy_weights)
# %%
observations = env.reset()
max_cycles = 10
avg_rewards = []
for step in range(max_cycles):

    obs = [jnp.array(observation).astype(jnp.float64) for observation in list(observations.values())]
    obs = jnp.array(obs)
    obs = jnp.reshape(obs, (20,1,84,84))
    cnn_output = conv_net(params, obs)

    key, subkey = random.split(key)
    values = value_function_head(vf_weights, cnn_output)
    log_prob, (actions, key) = policy_head(cnn_output, key, policy_weights)

    actions = {agent: int(actions[idx]) for idx, agent in zip(list(range(len(values))),env.agents)}
    observations, rewards, dones, infos = env.step(actions)#
    avg_rewards.append(np.mean(list(rewards.values())))
# %%
