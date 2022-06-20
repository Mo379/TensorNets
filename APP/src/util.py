import os
from datetime import timedelta
from time import sleep, time

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np

from tensorboardX import SummaryWriter

# Feature extractonesor CNN 
initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
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
      b_init=initializer, 
      name='NatureCNN_l1'
  )(x)
  x = jax.nn.relu(x)
  x = hk.Conv2D(
      64, (4,4), 
      stride=(2,2),
      padding='VALID', 
      with_bias=True, 
      w_init=initializer, 
      b_init=initializer, 
      name='NatureCNN_l2'
  )(x)
  x = jax.nn.relu(x)
  x = hk.Conv2D(
      64, (3,3), 
      stride=(1,1),
      padding='VALID', 
      with_bias=True, 
      w_init=initializer, 
      b_init=initializer, 
      name='NatureCNN_l3'
  )(x)
  x = jax.nn.relu(x)
  x = hk.Flatten(preserve_dims=1)(x)
  x = hk.Linear(
      512,
      with_bias=True, 
      w_init=initializer, 
      b_init=initializer, 
      name='NatureCNN_l4'
  )(x)
  x = jax.nn.relu(x)
  return x
def policy_net(x):
  policy_out = hk.nets.MLP([1], name='policy_net')(x)
  return policy_out
def value_net(x):
    value_out = hk.nets.MLP([1], name='value_net')(x)
    return value_out
class log_std(hk.Module):
  def __init__(self, deterministic=False, name=None):
    super().__init__(name=name)
    self.deterministic=deterministic
    self.rng = hk.PRNGSequence(0)
  def __call__(self, action_mean):
    log_std = hk.get_parameter("constant", shape=(1,), dtype=action_mean.dtype, init=initializer)
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









def my_model(x):
    features = feature_extractor(x)
    #
    values = value_net(features)
    action_mean = policy_net(features)
    #
    actions, sd= log_std(name='log_std')(action_mean)
    #
    return actions,values,sd

def my_actor(x):
    features = feature_extractor(x)
    #
    action_mean = policy_net(features)
    #
    actions, sd = log_std(name='log_std')(action_mean)
    #
    return actions,sd

def my_critic(x):
    features = feature_extractor(x)
    #
    values = value_net(features)
    #
    #
    return values







def my_TN_model(x):
    features = feature_extractor(x)
    pass

class TN_layer(hk.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
  def __call__(self, x):
    return x












def transfer_params(trained_params, model_params):
  transferred_dict = {}
  #log_std layer
  trained_log_std = 'log_std'
  log_std_constant = trained_params[trained_log_std].numpy()
  transferred_dict.update({ 'log_std': {
          'constant': jnp.round(log_std_constant,4)
      } 
  })

  #reset of the network
  trained_params.pop('log_std')
  trained_keys =  list(trained_params.keys())
  trained_keys = [trained_keys[i:i+2] for i in range(0, len(trained_keys), 2)]
  model_params.pop('log_std')
  model_keys = list(model_params.keys())
  desired_order_list = [0,1, 2, 3, 5, 4]
  model_keys = [model_keys[k] for k in desired_order_list]
  for i,keys in enumerate(zip(trained_keys,model_keys)):
    trained_keys, my_keys = keys
    #get layer params
    weights = trained_params[trained_keys[0]].numpy()
    bias = trained_params[trained_keys[1]].numpy()
    if i in [0,1,2]:
        weights = jnp.flip(weights.T)
        bias = jnp.flip(bias)
    if i in [3,4,5]:
        weights = jnp.flip(weights.T)
        bias = jnp.flip(bias)
    transferred_dict.update({
      my_keys: {
          'w': jnp.round(weights, 4),
          'b': jnp.round(bias,4)
      } 
    })
  return transferred_dict










def environment_setup():
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
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
    env = ss.frame_stack_v1(env, stack_size=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 
        1, 
        num_cpus=4, 
        base_class='stable_baselines3'
    )
    return env


class my_Trainer:
    """
    Trainer.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed=0,
        action_repeat=1,
        num_agent_steps=10 ** 6,
        eval_interval=10 ** 4,
        num_eval_episodes=10,
        save_params=True,
    ):
        assert num_agent_steps % action_repeat == 0
        assert eval_interval % action_repeat == 0

        # Envs.
        self.env = env
        self.env_test = env_test

        # Set seeds.
        self.env.seed(seed)
        self.env_test.seed(2 ** 31 - seed)

        # Algorithm.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.param_dir = os.path.join(log_dir, "param")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "summary"))

        # Other parameters.
        self.action_repeat = action_repeat
        self.num_agent_steps = num_agent_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.save_params = save_params

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Initialize the environment.
        state = self.env.reset()

        for step in range(1, self.num_agent_steps + 1):
            state = self.algo.step(self.env, state)

            if self.algo.is_update():
                print('learning')
                self.algo.update(self.writer)

            if step % self.eval_interval == 0:
                #self.evaluate(step)
                
                if self.save_params:
                    print('saving_params')
                    self.algo.save_params(os.path.join(self.param_dir, f"step{step}"))

        # Wait for the logging to be finished.
        sleep(2)

    def evaluate(self, step):
        total_return = 0.0
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            while not done:
                action = self.algo.select_action(state)
                state, reward, done, _ = self.env_test.step(action)
                total_return += reward

        # Log mean return.
        mean_return = total_return / self.num_eval_episodes
        # To TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step * self.action_repeat)
        # To CSV.
        self.log["step"].append(step * self.action_repeat)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to standard output.
        print(f"Num steps: {step * self.action_repeat:<6}   Return: {mean_return:<5.1f}   Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
