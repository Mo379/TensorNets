import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np



# Feature extractonesor CNN 
initializer = jnp.zeros #hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
initializer_bias = jnp.zeros
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






def policy_net(x):
  policy_out = hk.nets.MLP(
          [1], 
          with_bias=True, 
          w_init=initializer, 
          b_init=initializer_bias, 
          name='policy_net')(x)
  return policy_out






def value_net(x):
    value_out = hk.nets.MLP(
        [1], 
        with_bias=True, 
        w_init=initializer, 
        b_init=initializer_bias, 
        name='value_net')(x)
    return value_out





class log_std(hk.Module):
  def __init__(self, deterministic=True, name=None):
    super().__init__(name=name)
    self.deterministic=deterministic
    self.rng = hk.PRNGSequence(0)
  def __call__(self, action_mean):
    log_std = hk.get_parameter("constant", shape=(1,), dtype=action_mean.dtype, init=jnp.ones)
    key = next(self.rng)
    get_actions, get_log_prob = Normal(key, action_mean, log_std, sample_maxima=self.deterministic)
    #
    actions = get_actions()
    log_prob = get_log_prob(actions)
    actions = jnp.clip(actions,-1,1)
    return actions, log_std








def Normal(rng,mean, sd, sample_maxima):
    def random_sample(rng=rng,mean=mean,sd=sd, sample_maxima=sample_maxima):
        if sample_maxima:
            return mean
        x = mean + sd * jax.random.normal(rng, (1,))
        return x
    def log_prob(x,rng=rng,mean=mean,sd=sd):
        var = (sd ** 2)
        log_sd = jnp.log(sd) 
        return -((x - mean) ** 2) / (2 * var) - log_sd - jnp.log(jnp.sqrt(2 * jnp.pi))
    return random_sample, log_prob






class TN_layer(hk.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
  def __call__(self, x):
    return x
