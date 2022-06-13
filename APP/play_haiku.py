import os
from src.util import *
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
from pathlib import Path
import pickle


# Initialising the feature extractor
model_features= hk.transform(my_model)
rng = jax.random.PRNGKey(0)
examples = jax.random.normal(rng,(1,84,84,4))
model_features_params = model_features.init(rng, examples)
# Transfer parameters from the saved model
root= Path(__file__).resolve().parent.parent
_path = os.path.join(root,'pkls/models/policy')
model = PPO.load(_path)
trained_params = model.get_parameters()
transferred_params = transfer_params(trained_params['policy'], model_features_params)


_path_pkl = os.path.join(root,'pkls/models/haiku_transfer.pkl')
# Saving the objects:
print(_path_pkl, type(_path_pkl))
with open(_path_pkl, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(transferred_params, f)

# making haiku model
model= hk.transform(my_model)
rng = jax.random.PRNGKey(0)
examples = jax.random.normal(rng,(1,84,84,4))
#Loading haiku params
root= Path(__file__).resolve().parent.parent
file = open(_path_pkl, 'rb')
params = pickle.load(file)
file.close()
#

env = pistonball_v6.env()
env = ss.color_reduction_v0(env,mode='B')
env = ss.resize_v1(env, x_size=84,y_size=84)
env = ss.frame_stack_v1(env, 4)

if __name__ == '__main__':
    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        obs = obs.reshape((1,) + obs.shape)
        act= model.apply(params,rng,obs)[0][0] if not done else None
        env.step(act)
        env.render()
