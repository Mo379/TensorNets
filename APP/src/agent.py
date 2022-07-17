#system
#ML
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
from numbers import Real
import distrax
from jax import grad, jit, vmap
from jax import random
from jax.lax import cond, fori_loop
from jax.config import config


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
#tensornetwork components
#tensor Policy network
def tn_policy_net(x):
    x = TensorNet(
            32,1,
            True,
            name='policy_network_1'
            )(x)
    #normilsation and sigmoid activation
    x = jnp.log(x**2)
    x = jax.nn.sigmoid(x)
    return x
#value network
def tn_value_net(x):
    x = TensorNet(
            32,1,
            name='value_network_1'
            )(x)
    #x = jnp.log(jnp.trace(x)**2)
    x = jnp.log(x**2)
    x = jax.nn.sigmoid(x)
    return x
#defining the tensor network layer
class TensorNet(hk.Module):
    #
    def __init__(self, contraction_dim_size,output_dim_size,normalise=False,value_scan=True,name=None):
        super().__init__(name=name)
        #save params
        self.contraction_dim_size = contraction_dim_size
        self.output_dim_size = output_dim_size
        self.normalise = normalise
    #
    def __call__(self, inputs):
        mps_params = hk.get_parameter(
            "mps", 
            #dims(agent,feature_vec,contraction,output)
            shape=(
                inputs.shape[1],
                inputs.shape[0],
                self.output_dim_size,
                self.contraction_dim_size,
                self.contraction_dim_size
            ), 
            dtype=inputs.dtype, 
            init=initializer
        )
        #forwards pass here
        output = self.tensor_scan(inputs,mps_params)
        #
        if self.normalise:
            output = self._normalise(output)
        return output
    def _normalise(self,x):
        return x
    #tensor scan function
    def tensor_scan(self,embedding_vectors, mps_params):
        env = jnp.tensordot(embedding_vectors[0], mps_params, axes=((0),(0)))
        val = (env, embedding_vectors, mps_params)
        val = fori_loop(1, len(embedding_vectors), self._tensor_step, val)
        return jnp.trace(val[0].T).T
    #scan helper function
    def _tensor_step(self,step, val):
        env, embedding_vectors, mps_params = val
        mat = jnp.tensordot(embedding_vectors[step], mps_params, axes=((0),(0)))
        env = jnp.matmul(env, mat)
        env = jax.nn.sigmoid(env)
        return env, embedding_vectors, mps_params
#distribution layer
class log_std(hk.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
  def __call__(self, actions_means):
    sd = hk.get_parameter("constant", shape=(1,), dtype=actions_means.dtype, init=jnp.zeros)
    #
    return actions_means, sd






#Models
#actor critic model
def my_model(x):
    features = feature_extractor(x)
    #
    values = value_net(features)
    action_mean = policy_net(features)
    #
    actions, log_sd= log_std(name='log_std')(action_mean)
    #
    return actions,values,log_sd

def my_model_tensornet(x):
    #get input features
    x = feature_extractor(x)
    x = hk.Linear(
        32,
        with_bias=True, 
        w_init=initializer, 
        b_init=initializer_bias, 
        name='Tensornet_dim_reduction'
    )(x)
    x = jax.nn.relu(x)
    #
    values = tn_value_net(x)
    action_mean = tn_policy_net(x)
    actions, sd = log_std(name='log_std')(action_mean)
    return actions,values.reshape((-1,1)), sd
