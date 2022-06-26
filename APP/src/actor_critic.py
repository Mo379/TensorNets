import os
from abc import abstractmethod
from functools import partial
from typing import List

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .base_algoirithm import OnPolicyAlgorithm
from .saving import load_params, save_params     
from .fake_input import fake_action, fake_state 


class ActorCriticMixIn:
    """ MixIn for Actor-Critic algorithms. """

    def __init__(self):
        # If _loss_critic() method uses random key or not.
        if not hasattr(self, "use_key_critic"):
            self.use_key_critic = False
        # If _loss_actor() method uses random key or not.
        if not hasattr(self, "use_key_actor"):
            self.use_key_actor = False

    @property
    def kwargs_critic(self):
        return {"key": next(self.rng)} if self.use_key_critic else {}

    @property
    def kwargs_actor(self):
        return {"key": next(self.rng)} if self.use_key_actor else {}

    def select_action(self, state):
        action = self._select_action(self.params_policy, state[None, ...])
        return np.array(action)

    @abstractmethod
    def _select_action(self, params_policy, state):
        pass

    @abstractmethod
    def _explore(self, params_policy, state, key):
        pass

    def save_params(self, save_dir):
        save_params(self.params_policy, os.path.join(save_dir, "params_policy.npz"))

    def load_params(self, save_dir):
        self.params_policy= load_params(os.path.join(save_dir, "params_policy.npz"))


class OnPolicyActorCritic(ActorCriticMixIn, OnPolicyAlgorithm):
    """
    Base class for on-policy Actor-Critic algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
        buffer_size,
        batch_size,
    ):
        ActorCriticMixIn.__init__(self)
        OnPolicyAlgorithm.__init__(
            self,
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
        # Define fake input for policy.
        if not hasattr(self, "fake_args_policy"):
            self.fake_args_policy= (fake_state(state_space),)
        # Define fake input for critic.
        if not hasattr(self, "fake_args_critic"):
            self.fake_args_critic = (fake_state(state_space),)
        # Define fake input for actor.
        if not hasattr(self, "fake_args_actor"):
            self.fake_args_actor = (fake_state(state_space),)

    def is_update(self):
        return self.agent_step % self.buffer_size == 0

    def explore(self, state):
        action, log_pi = self._explore(self.params_policy, state, next(self.rng))
        return np.array(action), np.array(log_pi)
