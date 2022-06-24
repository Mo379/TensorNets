from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax as optix

from .actor_critic import OnPolicyActorCritic
from .distribution import evaluate_gaussian_and_tanh_log_prob, reparameterize_gaussian_and_tanh
from .optim import optimize
import wandb


class PPO(OnPolicyActorCritic):
    name = "PPO"

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
        fn_actor,
        fn_critic,
        lr_actor,
        lr_critic,
        epoch_ppo,
        clip_eps,
        lambd,
    ):
        assert buffer_size % batch_size == 0
        super(PPO, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
        # Critic.
        self.critic = hk.without_apply_rng(hk.transform(fn_critic))
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), *self.fake_args_critic)
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)

        # Actor.
        self.actor = hk.without_apply_rng(hk.transform(fn_actor))
        self.params_actor = self.params_actor_target = self.actor.init(next(self.rng), *self.fake_args_actor)
        opt_init, self.opt_actor = optix.adam(lr_actor)
        self.opt_state_actor = opt_init(self.params_actor)

        # Other parameters.
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.max_grad_norm = max_grad_norm
        self.idxes = np.arange(buffer_size)

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        mean, _ = self.actor.apply(params_actor, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        #no randomisation done by actor apply!!
        mean, log_std = self.actor.apply(params_actor, state)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, True)

    def update(self, wandb_run):
        #get buffer values
        print('getting_instance')
        outputs = self.buffer.get()
        actor_first_output= []
        for output in outputs:
            actor_first_output.append(jnp.swapaxes(output, 0,1))
        states, actions, rewards, dones, log_pi_olds, next_states = actor_first_output
        # Calculate gamma-retwandb_runurns and GAEs.
        print('Calculate gamma-returns and GAEs.')
        gae, target = self.calculate_gae(
            params_critic=self.params_critic,
            actors_states=states,
            actors_rewards=rewards,
            actors_dones=dones,
            actors_next_states=next_states,
        )
        print('done gae got')
        exit()
        #
        for i_count in range(self.epoch_ppo):
            print(f"epoch {i_count}")
            np.random.shuffle(self.idxes)
            for start in range(0, self.buffer_size, self.batch_size):
                self.learning_step += 1
                idx = self.idxes[start : start + self.batch_size]
                # Update critic.
                self.opt_state_critic, self.params_critic, loss_critic, _ = optimize(
                    self._loss_critic,
                    self.opt_critic,
                    self.opt_state_critic,
                    self.params_critic,
                    self.max_grad_norm,
                    state=state[idx],
                    target=target[idx],
                )
                # Update actor.
                self.opt_state_actor, self.params_actor, loss_actor, _ = optimize(
                    self._loss_actor,
                    self.opt_actor,
                    self.opt_state_actor,
                    self.params_actor,
                    self.max_grad_norm,
                    state=state[idx],
                    action=action[idx],
                    log_pi_old=log_pi_old[idx],
                    gae=gae[idx],
                )
        #log the losses
        wandb.log({"loss/critic": np.array(loss_critic)})
        wandb.log({"loss/actor": np.array(loss_actor)})

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        target: np.ndarray,
    ) -> jnp.ndarray:
        return jnp.square(target - self.critic.apply(params_critic, state)).mean(), None

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        log_pi_old: np.ndarray,
        gae: jnp.ndarray,
    ) -> jnp.ndarray:
        # Calculate log(\pi) at current policy.
        mean, log_std = self.actor.apply(params_actor, state)
        log_pi = evaluate_gaussian_and_tanh_log_prob(mean, log_std, action)
        # Calculate importance ratio.
        ratio = jnp.exp(log_pi - log_pi_old)
        loss_actor1 = -ratio * gae
        loss_actor2 = -jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gae
        loss_actor = jnp.maximum(loss_actor1, loss_actor2).mean()
        return loss_actor, None

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self,
        params_critic: hk.Params,
        actors_states: np.ndarray,
        actors_rewards: np.ndarray,
        actors_dones: np.ndarray,
        actors_next_states: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        actors_gae = []
        actors_targets = []
        for state, reward, done, next_state in zip(actors_states,actors_rewards,actors_dones,actors_next_states): 
            # Current and next value estimates.
            value = jax.lax.stop_gradient(self.critic.apply(params_critic, state))
            next_value = jax.lax.stop_gradient(self.critic.apply(params_critic, next_state))
            # Calculate TD errors.
            delta = reward + self.gamma * next_value * (1.0 - done) - value
            # Calculate GAE recursively from behind.
            gae = [delta[-1]]
            for t in jnp.arange(self.buffer_size - 2, -1, -1):
                gae.insert(0, delta[t] + self.gamma * self.lambd * (1 - done[t]) * gae[0])
            gae = jnp.array(gae)
            actors_gae.append((gae - gae.mean()) / (gae.std() + 1e-8))
            actors_targets.append(gae + value)
        print('done internal gae loop')
        return actors_gae, actors_targets
