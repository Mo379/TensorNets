import os
from functools import partial
from typing import Tuple
import pickle
#
import haiku as hk
from haiku import PRNGSequence
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import distrax
#
from .util import RolloutBuffer
import wandb


class PPO():
    name = "PPO"

    # Init function
    def __init__(
        self,
        # seed and root
        seed,
        root,
        # models and model hyperparams
        fn_policy,
        lr_policy,
        # algorithm hyper params
        max_grad_norm,
        gamma,
        clip_eps,
        lambd,
        ent_coef,
        vf_coef,
        # env hyperparams
        state_space,
        action_space,
        # training length hyperparams
        num_agent_steps,
        buffer_size,
        batch_size,
        epoch_ppo,
    ):
        # seed and root
        self.seed = seed
        # rng
        np.random.seed(self.seed)
        self.rng = PRNGSequence(self.seed)
        self.root = root
        # Policy.
        self.policy = hk.without_apply_rng(hk.transform(fn_policy))
        self.params_policy = self.policy.init(
                next(self.rng),
                np.random.normal(size=(20, 84, 84, 3))
            )
        # Other parameters.
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.ent_coef = vf_coef
        self.vf_coef = vf_coef
        # state and action spaces
        self.state_space = state_space
        self.action_space = action_space
        # init buffer
        self.num_agent_steps = num_agent_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch_ppo = epoch_ppo
        # optimiser
        opt_init, self.opt_policy = optax.adam(lr_policy)
        self.opt_state_policy = opt_init(self.params_policy)
        # step counts
        self.idxes = np.arange(buffer_size)
        self.buffer = RolloutBuffer()
        self.optax_zero_init, self.optax_zero_apply = optax.set_to_zero()
        self.max_abs_reward = jnp.inf
        self.clip_value = True

    # Step function
    def step(self, env, state, done):
        # reset env if done playing
        if done.all():
            state = env.reset()
        # explore the environment
        action, log_prob = self._explore(
                self.params_policy,
                state,
                next(self.rng)
            )
        # move the environment forwards
        next_state, reward, done, _ = env.step(action)
        # save the relevant information into the buffer
        self.buffer.append(state, action, log_prob, reward, done, next_state)
        # retrun state and done
        return next_state, done

    # Update function
    def update(self, wandb_run):
        # get buffer items and calculate state value
        print('getting_instance')
        # get all data
        outputs = self.buffer.get()
        #
        actor_first_output = []
        for output in outputs:
            output = jnp.array(output)
            output = jnp.swapaxes(output, 0, 1)
            actor_first_output.append(output)
        # unpack data
        A_states, A_actions, A_log_pi_olds, A_rewards, A_dones, A_next_states \
            = actor_first_output
        A_discounts = A_dones*self.lambd

        # calculate values after swapping axes
        def get_play_values(params, obs):
            o = jnp.swapaxes(obs, 0, 1)
            behavior_values = [
                    self._get_value(self.params_policy, os) for os in o
                ]
            behavior_values = jnp.array(behavior_values)
            behavior_values = jnp.swapaxes(behavior_values, 0, 1)
            behavior_values = jnp.squeeze(behavior_values)
            return behavior_values

        A_values = get_play_values(self.params_policy, A_states)
        # getting GAEs and targets
        print('Calculating gaes for all agents')
        agent_wise_gae = jax.vmap(self.gae_advantages, in_axes=0)
        A_gaes, A_targets = agent_wise_gae(
            A_rewards,
            A_discounts,
            A_values,
        )
        A_gaes = jnp.array(A_gaes)
        A_targets = jnp.array(A_targets)
        # align data discarding last step
        A_states, A_actions, A_log_pi_olds, A_rewards, A_dones, \
            A_next_states, A_values = jax.tree_map(
                    lambda x: x[:, :-1],
                    (
                        A_states,
                        A_actions,
                        A_log_pi_olds,
                        A_rewards,
                        A_dones,
                        A_next_states,
                        A_values
                    )
                )
        #
        A_dataframe = (
                A_states,
                A_actions,
                A_log_pi_olds,
                A_rewards,
                A_dones,
                A_next_states,
                A_gaes,
                A_targets,
                A_values
            )
        A_shuffler = np.random.permutation(A_states.shape[1])
        # flatten data fuse first two dimentions (agent and play-timesetp)
        A_n_outputs = []
        for data in A_dataframe:
            # reshaped = jnp.reshape(data,(-1,)+data.shape[2:])
            data = jnp.swapaxes(data, 0, 1)
            reshaped = data[A_shuffler]
            A_n_outputs.append(reshaped)
        # create n batches matching batch size
        idxes = np.arange(len(A_n_outputs[1]))
        assert len(idxes) % self.batch_size == 0
        A_n_outputs = jax.tree_map(
                lambda x:
                x.reshape((-1, self.batch_size, ) + x.shape[1:]),
                A_n_outputs
            )
        #
        self.grad_fn = jax.grad(self.loss, has_aux=True)
        # Repeat training for the given number of epoch, taking a random
        # permutation for every epoch.
        print('updating params')
        (self.params_policy, self.opt_state_policy, _), (metrics, test) = \
            jax.lax.scan(
                self._model_update_epoch,
                (
                    self.params_policy,
                    self.opt_state_policy,
                    A_n_outputs
                ),
                None,
                length=self.epoch_ppo
            )
        #
        metrics = jax.tree_map(jnp.mean, metrics)
        metrics['norm_params'] = optax.global_norm(self.params_policy)
        metrics['rewards_mean'] = jnp.mean(
            jnp.abs(jnp.mean(A_rewards, axis=(0, 1))))
        metrics['rewards_std'] = jnp.std(A_rewards, axis=(0, 1))
        #
        if wandb_run:
            # log the losses
            wandb.log({"loss/critic": np.array(metrics['loss_value'])})
            wandb.log({"loss/actor": np.array(metrics['loss_policy'])})
            wandb.log({"loss/entropy": np.array(metrics['loss_entropy'])})
            wandb.log({"loss/ppo_total_loss": np.array(metrics['loss_total'])})
            wandb.log({"reward/mean": np.array(metrics['rewards_mean'])})
            wandb.log({"reward/std": np.array(metrics['rewards_std'])})
        self.buffer.clear()

    # save and load params
    def fn_save_params(self, path):
        """
        Save parameters.
        """
        path = os.path.join(path, "params_policy.pkl")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.params_policy, f)

    # load params
    def fn_load_params(self, path):
        """
        Load parameters.
        """
        path = os.path.join(path, "haiku_transfer.pkl")
        file = open(path, 'rb')
        params = pickle.load(file)
        return params

    # Model invocations
    @partial(jax.jit, static_argnums=0)
    def _get_value(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        _, value, _ = self.policy.apply(params_policy, state)
        return value

    # action selection
    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action, _, log_std = self.policy.apply(params_policy, state)
        dist = distrax.MultivariateNormalDiag(
                action,
                jnp.ones_like(action)*jnp.exp(log_std)
            )
        action = jnp.clip(action, -1, 1)
        #
        log_prob = dist.log_prob(action)
        return action, log_prob

    # explore
    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # randomisation done by policy apply!!
        action, v, sd = self.policy.apply(params_policy, state)
        dist = distrax.MultivariateNormalDiag(
                action, jnp.ones_like(action)*jnp.exp(sd)
            )
        actions = dist.sample(seed=key)
        actions = jnp.clip(actions, -1, 1)
        log_prob = dist.log_prob(actions)
        return actions, log_prob

    # proxy model invocations (only use this outside of this class)
    def explore(self, state):
        action, log_prob = self._explore(
                self.params_policy, state, next(self.rng))
        return action, log_prob

    # action selection
    def select_action(self, state):
        action, log_prob = self._select_action(self.params_policy, state)
        return np.array(action), np.array(log_prob)

    # generalised advantage estimation
    @partial(jax.jit, static_argnums=0)
    def gae_advantages(
            self,
            rewards: jnp.array,
            discounts: jnp.array,
            values: jnp.array
    ) -> Tuple[jnp.ndarray, jnp.array]:
        """Uses truncated GAE to compute advantages."""
        # Apply reward clipping.
        rewards = jnp.clip(
                rewards,
                -self.max_abs_reward,
                self.max_abs_reward
            )
        advantages = rlax.truncated_generalized_advantage_estimation(
                rewards[:-1],
                discounts[:-1],
                self.lambd,
                values
            )
        advantages = jax.lax.stop_gradient(advantages)
        # Exclude the bootstrap value
        target_values = values[:-1] + advantages
        target_values = jax.lax.stop_gradient(target_values)
        return advantages, target_values

    # PPO loss function, surragoat loss
    @partial(jax.jit, static_argnums=0)
    def loss(
        self,
        params,
        observations,
        actions,
        behaviour_log_probs,
        target_values,
        advantages,
        behavior_values
    ):
        """Surrogate loss using clipped probability ratios."""
        #
        batch_outputs = [self.policy.apply(params, o) for o in observations]
        means = jnp.array([b[0] for b in batch_outputs])
        values = jnp.array([b[1] for b in batch_outputs])
        sd = jnp.array([b[2] for b in batch_outputs])
        actions = actions
        #
        dists = [
                distrax.MultivariateNormalDiag(
                    m,
                    jnp.ones_like(m)*jnp.exp(s)
                )
                for m, s in zip(means, sd)
            ]
        log_probs = jnp.array(
                [
                    dist.log_prob(a) 
                    for dist, a in zip(dists, actions)
                ]
            )
        entropy = jnp.array([dist.entropy() for dist in dists])

        # reshaping
        values = values.reshape((-1, 1))
        behavior_values = behavior_values.reshape((-1, 1))
        actions = actions.reshape((-1, 1))
        target_values = target_values.reshape((-1,))
        behaviour_log_probs = behaviour_log_probs.reshape((-1,))
        log_probs = log_probs.reshape((-1,))
        entropy = entropy.reshape((-1,))
        advantages = advantages.reshape([-1])
        # Compute importance sampling weights:
        # current policy / behavior policy.
        rhos = jnp.exp(log_probs - behaviour_log_probs)
        policy_loss = rlax.clipped_surrogate_pg_loss(
                rhos,
                advantages,
                self.clip_eps
            )
        # Value function loss. Exclude the bootstrap value
        unclipped_value_error = target_values - values
        unclipped_value_loss = unclipped_value_error ** 2
        if self.clip_value:
            # Clip values to reduce variablility during critic training.
            clipped_values = behavior_values + jnp.clip(
                values - behavior_values, -self.clip_eps,
                self.clip_eps)
            clipped_value_error = target_values - clipped_values
            clipped_value_loss = clipped_value_error ** 2
            value_loss = jnp.mean(
                    jnp.fmax(
                        unclipped_value_loss,
                        clipped_value_loss
                    )
                )
        else:
            # For Mujoco envs clipping hurts a lot. Evidenced by Figure 43 in
            # https://arxiv.org/pdf/2006.05990.pdf
            value_loss = jnp.mean(unclipped_value_loss)
        # Entropy regulariser.
        entropy_loss = -jnp.mean(entropy)
        total_loss = (
                policy_loss +
                value_loss * self.vf_coef +
                entropy_loss * self.ent_coef
            )
        return total_loss, {
            'loss_total': total_loss,
            'loss_policy': policy_loss,
            'loss_value': value_loss,
            'loss_entropy': entropy_loss,
        }

    # Update function
    def _model_update_epoch(
        self,
        carry,
        unused_t,
    ):
        """Performs model updates based on one epoch of data."""
        params, opt_state, batch = carry
        #
        (params, opt_state), (metrics, test) = jax.lax.scan(
                self._model_update_minibatch,
                (params, opt_state),
                batch
            )
        return (params, opt_state, batch), (metrics, test)

    # update functions
    def _model_update_minibatch(
        self,
        carry,
        minibatch,
    ):
        """Performs model update for a single minibatch."""
        params, opt_state = carry
        b_states, b_actions, b_log_pi_olds, _, _, \
            _, b_gaes, b_targets, b_values = minibatch
        # Normalize advantages at the minibatch level before using them.
        advantages = ((b_gaes -
                       jnp.mean(b_gaes, axis=0)) /
                      (jnp.std(b_gaes, axis=0) + 1e-8))
        gradients, metrics = self.grad_fn(
                params,
                b_states,
                b_actions,
                b_log_pi_olds,
                b_targets,
                advantages,
                b_values
            )
        # Apply updates
        opt_state, _ = self.optax_zero_apply(
                opt_state,
                None
            )
        updates, opt_state = self.opt_policy(gradients, opt_state)
        params = optax.apply_updates(params, updates)

        metrics['norm_grad'] = optax.global_norm(gradients)
        metrics['norm_updates'] = optax.global_norm(updates)
        test = gradients
        return (params, opt_state), (metrics, test)
