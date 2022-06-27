#system
#ML
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
from numbers import Real


#model building blocks
#initialisers
initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
initializer_bias = jnp.zeros
#feature extractor
def feature_extractor(x):
  x = jnp.array(x, dtype=jnp.float32)
  x = x/255.0
  if len(x.shape) == 5:
      x = jnp.squeeze(x)
  x = hk.Conv2D(
      32, (8,8), 
      stride=(4,4), 
      padding='VALID', 
      with_bias=True, 
      w_init=initializer, 
      b_init=initializer_bias, 
      name='NatureCNN_l1'
  )(x)
  x = jax.nn.relu(x)
  x = hk.Conv2D(
      64, (4,4), 
      stride=(2,2),
      padding='VALID', 
      with_bias=True, 
      w_init=initializer, 
      b_init=initializer_bias, 
      name='NatureCNN_l2'
  )(x)
  x = jax.nn.relu(x)
  x = hk.Conv2D(
      64, (3,3), 
      stride=(1,1),
      padding='VALID', 
      with_bias=True, 
      w_init=initializer, 
      b_init=initializer_bias, 
      name='NatureCNN_l3'
  )(x)
  x = jax.nn.relu(x)
  x = hk.Flatten(preserve_dims=1)(x)
  x = hk.Linear(
      512,
      with_bias=True, 
      w_init=initializer, 
      b_init=initializer_bias, 
      name='NatureCNN_l4'
  )(x)
  x = jax.nn.relu(x)
  return x
#Policy network
def policy_net(x):
  policy_out = hk.nets.MLP(
          [1], 
          with_bias=True, 
          w_init=initializer, 
          b_init=initializer_bias, 
          name='policy_net')(x)
  return policy_out
#value network
def value_net(x):
    value_out = hk.nets.MLP(
        [1], 
        with_bias=True, 
        w_init=initializer, 
        b_init=initializer_bias, 
        name='value_net')(x)
    return value_out
#distribution layer
class log_std(hk.Module):
  def __init__(self, deterministic=True, name=None):
    super().__init__(name=name)
    self.deterministic=deterministic
    self.rng = hk.PRNGSequence(0)
  def __call__(self, action_mean):
    log_std = hk.get_parameter("constant", shape=(1,), dtype=action_mean.dtype, init=jnp.ones)
    key = next(self.rng)
    get_actions, get_log_prob = 0,1
    #
    actions = get_actions()
    log_prob = get_log_prob(actions)
    actions = jnp.clip(actions,-1,1)
    return actions, log_std
#Tensor network layer
class TN_layer(hk.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
  def __call__(self, x):
    return x









#Models
#actor critic model
def my_model(x, deterministic=True):
    features = feature_extractor(x)
    #
    values = value_net(features)
    action_mean = policy_net(features)
    #
    actions, sd= log_std(name='log_std', deterministic=deterministic)(action_mean)
    #
    return actions,values,sd
#actor model
def my_actor(x, deterministic=True):
    features = feature_extractor(x)
    #
    action_mean = policy_net(features)
    #
    actions, sd = log_std(name='log_std', deterministic=deterministic)(action_mean)
    #
    return actions,sd
#critic model
def my_critic(x):
    features = feature_extractor(x)
    #
    values = value_net(features)
    #
    return values
# tensornetwork actor critic model
def my_TN_model(x):
    features = feature_extractor(x)
    pass
