from abc import ABC, abstractmethod
from functools import partial

import jax
import numpy as np
from gym.spaces import Box
from haiku import PRNGSequence

from rollout_buffer import RolloutBuffer
from optim import soft_update


class BaseAlgorithm(ABC):
    """
    Base class for algorithms.
    """

    name = None

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
    ):
        np.random.seed(seed)
        self.rng = PRNGSequence(seed)

        self.agent_step = 0
        self.episode_step = 0
        self.learning_step = 0
        self.num_agent_steps = num_agent_steps
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.discrete_action = False if type(action_space) == Box else True

    def get_mask(self, env, done):
        return done if self.episode_step != 125 or self.discrete_action else False

    def get_key_list(self, num_keys):
        return [next(self.rng) for _ in range(num_keys)]

    @abstractmethod
    def is_update(self):
        pass

    @abstractmethod
    def step(self, env, state):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def update(self, writer):
        pass

    @abstractmethod
    def save_params(self, save_dir):
        pass

    @abstractmethod
    def load_params(self, save_dir):
        pass

    def __str__(self):
        return self.name


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    Base class for on-policy algorithms.
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
        super(OnPolicyAlgorithm, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
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

    def step(self, env, state):
        self.agent_step += 1
        self.episode_step += 1

        action, log_pi = self.explore(state)
        action = np.clip(action,-1,1)
        next_state, reward, done, _ = env.step(action)
        mask = self.get_mask(env, done)
        if done.any() == False:
            for i in range(0,len(action)):
                self.buffer.append(state[i], action[i], reward[i], mask[i], log_pi[i], next_state[i])

        if done.any():
            self.episode_step = 0
            next_state = env.reset()

        return next_state
