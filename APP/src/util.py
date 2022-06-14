import os
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np

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
  return x
def policy_net(x):
  policy_out = hk.nets.MLP([1], name='policy_net')(x)
  return policy_out
def value_net(x):
    value_out = hk.nets.MLP([1], name='value_net')(x)
    return value_out
def Normal(rng,mean, sd):
    def random_sample(rng=rng,mean=mean,sd=sd):
        x = mean + sd * jax.random.normal(rng, (1,))
        return x
    def log_prob(x,rng=rng,mean=mean,sd=sd):
        var = (sd ** 2)
        log_sd = jnp.log(sd) 
        return -((x - mean) ** 2) / (2 * var) - log_sd - jnp.log(jnp.sqrt(2 * jnp.pi))
    return random_sample, log_prob
def my_model(x):
    features = feature_extractor(x)
    #
    values = value_net(features)
    action_mean = policy_net(features)
    #
    actions, log_prob = log_std(name='log_std')(action_mean)
    #
    return actions,values,log_prob

def my_TN_model(x):
    features = feature_extractor(x)
    pass

class TN_layer(hk.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
  def __call__(self, x):
    return x




class log_std(hk.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
  def __call__(self, action_mean):
    log_std = hk.get_parameter("constant", shape=(1,), dtype=action_mean.dtype, init=jnp.ones)
    key = hk.next_rng_key()
    get_actions, get_log_prob = Normal(key, action_mean, log_std)
    #
    actions = get_actions()
    log_prob = get_log_prob(actions)
    return actions, log_prob










def transfer_params(trained_params, model_params):
  transferred_dict = {}
  #log_std layer
  trained_log_std = 'log_std'
  log_std_constant = trained_params[trained_log_std].numpy()
  transferred_dict.update({ 'log_std': {
          'constant': log_std_constant
      } 
  })

  #reset of the network
  trained_params.pop('log_std')
  trained_keys =  list(trained_params.keys())
  trained_keys = [trained_keys[i:i+2] for i in range(0, len(trained_keys), 2)]
  model_params.pop('log_std')
  model_keys = list(model_params.keys())
  for trained_keys, my_keys in zip(trained_keys,model_keys):
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
        n_pistons=20,
        time_penalty=-0.3,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=200
    )
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84,y_size=84)
    env = ss.frame_stack_v1(env, stack_size=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 
        16, 
        num_cpus=8, 
        base_class='stable_baselines3'
    )
    return env

