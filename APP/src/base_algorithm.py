#system
import os
from abc import abstractmethod
from functools import partial
from typing import List
#ML
from abc import ABC, abstractmethod
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku import PRNGSequence
import distrax
#Env
from gym.spaces import Box
#local
from .util import load_params, save_params,RolloutBuffer, fake_state 


#Base algorithm class
class BaseAlgorithm(ABC):
    """
    Base class for algorithms.
    """
    name = None
    #
    def __init__(
        self,
        #seed and root
        seed,
        root,
        #hyper paras
        num_agent_steps,
        state_space,
        action_space,
        max_grad_norm,
        gamma,
    ):
        #seed and root
        np.random.seed(seed)
        self.rng = PRNGSequence(seed)
        self.root = root
        #
        self.agent_step = 0
        self.episode_step = 0
        self.learning_step = 0
        self.num_agent_steps = num_agent_steps
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.discrete_action = False if type(action_space) == Box else True
    #
    def get_mask(self, env, done):
        #env._max_episode_steps
        return done if self.episode_step != 125 or self.discrete_action else False
    #
    def get_key_list(self, num_keys):
        return [next(self.rng) for _ in range(num_keys)]
    #
    @abstractmethod
    def is_update(self):
        pass
    #
    @abstractmethod
    def step(self, env, state):
        pass
    #
    @abstractmethod
    def select_action(self, state):
        pass
    #
    @abstractmethod
    def explore(self, state):
        pass
    #
    @abstractmethod
    def update(self, writer):
        pass
    #
    @abstractmethod
    def save_params(self, save_dir):
        pass
    #
    @abstractmethod
    def load_params(self, save_dir):
        pass
    #
    def __str__(self):
        return self.name









#On ploicy algorithm
class OnPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for on-policy algorithms.
    """

    def __init__(
        self,
        #seed and root
        seed,
        root,
        #hyperparams
        num_agent_steps,
        state_space,
        action_space,
        max_grad_norm,
        gamma,
        buffer_size,
        batch_size,
    ):
        super(OnPolicyAlgorithm, self).__init__(
            #seed and root
            seed=seed,
            root=root,
            #hyperparams
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
        )
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
        )
        self.discount = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def step(self, env, state, done):
        #
        self.agent_step += 1
        self.episode_step += 1
        #
        if self.agent_step > 1 and done.any():
            self.episode_step = 0
            state = env.reset()
        #
        action, log_prob = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        done_mask = self.get_mask(env, done)
        #
        self.buffer.append(state, action, reward, done_mask, log_prob, next_state)
        #
        return next_state, done








#Mixin for the actor critic
class ActorCriticMixIn:
    """ MixIn for Actor-Critic algorithms. """
    #init function 
    def __init__(self):
        # If _loss_critic() method uses random key or not.
        if not hasattr(self, "use_key_policy"):
            self.use_key_policy = False
    #advance by taking a deterministic action (exploitation)
    def select_action(self, state):
        action,log_prob = self._select_action(self.params_policy, state)
        return np.array(action), np.array(log_prob)
    #advance by taking a deterministic action (exploitation)
    @abstractmethod
    def _select_action(self, params_policy, state):
        pass
    #advance by taking a random action (exploration)
    @abstractmethod
    def _explore(self, params_policy, state, key):
        pass
    #Saving the parameters 
    def save_params(self, save_dir):
        save_params(self.params_policy, os.path.join(save_dir, "params_policy.pkl"))
    #Loading the parameters
    def load_params(self, save_dir):
        self.params_policy= load_params(os.path.join(save_dir, "params_policy.pkl"))











#Onpolicy actor critic algorithm
class OnPolicyActorCritic(ActorCriticMixIn, OnPolicyAlgorithm):
    """
    Base class for on-policy Actor-Critic algorithms.
    """

    def __init__(
        self,
        #seed and root
        seed,
        root,
        #hyperparams
        num_agent_steps,
        state_space,
        action_space,
        max_grad_norm,
        gamma,
        buffer_size,
        batch_size,
    ):
        ActorCriticMixIn.__init__(self)
        OnPolicyAlgorithm.__init__(
            self,
            #seed and root
            seed=seed,
            root=root,
            #hyper params
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
        # Define fake input for policy.
        if not hasattr(self, "fake_args_policy"):
            self.fake_args_policy= (fake_state(state_space),)
    #
    def is_update(self):
        return self.agent_step % self.buffer_size == 0
    #
    def explore(self, state):
        action, log_prob = self._explore(self.params_policy, state, next(self.rng))
        return action, log_prob
