import os
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np

#self.cnn = nn.Sequential(
#    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#    nn.ReLU(),
#    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#    nn.ReLU(),
#    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#    nn.ReLU(),
#    nn.Flatten(),
#}
class log_std(hk.Module):

  def __init__(self, name=None):
    super().__init__(name=name)

  def __call__(self, x):
    w = hk.get_parameter("constant", shape=(1,), dtype=x.dtype, init=jnp.ones)
    x = x*w
    return x
# Feature extractor CNN 
def feature_extractor(x):
  x = x/255
  x = hk.Conv2D(
      32, 8, 
      stride=4, 
      padding='VALID', 
      with_bias=True, 
      w_init=None, 
      b_init=None, 
      name='NatureCNN_l1'
  )(x)
  x = jax.nn.relu(x)
  x = hk.Conv2D(
      64, 4, 
      stride=2,
      padding='VALID', 
      with_bias=True, 
      w_init=None, 
      b_init=None, 
      name='NatureCNN_l2'
  )(x)
  x = jax.nn.relu(x)
  x = hk.Conv2D(
      64, 3, 
      stride=1,
      padding='VALID', 
      with_bias=True, 
      w_init=None, 
      b_init=None, 
      name='NatureCNN_l3'
  )(x)
  x = jax.nn.relu(x)
  x = hk.Flatten(preserve_dims=1)(x)
  x = hk.Linear(
      512,
      with_bias=True, 
      w_init=None, 
      b_init=None, 
      name='NatureCNN_l4'
  )(x)
  policy_out = hk.nets.MLP([1], name='policy_net')(x)
  policy_out = log_std(name='log_std')(policy_out)
  value_out = hk.nets.MLP([1], name='value_net')(x)
  return policy_out,value_out





def transfer_params(trained_params, model_params):
  trained_keys =  list(trained_params.keys())[1:]
  trained_keys = [trained_keys[i:i+2] for i in range(0, len(trained_keys), 2)]
  model_keys = list(model_params.keys())
  transferred_dict = {}

  #log_std layer
  trained_log_std = list(trained_params.keys())[0]
  log_std_constant = trained_params[trained_log_std].numpy()
  transferred_dict.update({ model_keys[0]: {
          'constant': log_std_constant
      } 
  })
  #reset of the network
  for trained_keys, my_keys in zip(trained_keys,model_keys[1:]):
    #get layer params
    weights = trained_params[trained_keys[0]].numpy().T
    bias = trained_params[trained_keys[1]].numpy()
    transferred_dict.update({
      my_keys: {
          'w': weights,
          'b': bias
      } 
    })
  return transferred_dict

def environment_setup():
    env = pistonball_v6.parallel_env(
        n_pistons=30,
        time_penalty=-1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125
    )
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84,y_size=84)
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 
        4, 
        num_cpus=2, 
        base_class='stable_baselines3'
    )
    return env

