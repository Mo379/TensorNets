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
from src.network_components import *
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
        fn_policy,
        fn_actor,
        fn_critic,
        lr_policy,
        lr_actor,
        lr_critic,
        epoch_ppo,
        clip_eps,
        lambd,
        ent_coef,
        vf_coef,
    ):
        #assert buffer_size % batch_size == 0
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
        # Policy.
        self.policy= hk.without_apply_rng(hk.transform(fn_policy))
        self.params_policy = self.params_policy_target = self.policy.init(next(self.rng), *self.fake_args_policy)
        #self.load_params('logs/pkls')
        opt_init, self.opt_policy= optix.adam(lr_policy)
        self.opt_state_policy= opt_init(self.params_policy)
        # Other parameters.
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.ent_coef=vf_coef
        self.vf_coef=vf_coef
        self.max_grad_norm = max_grad_norm
        self.idxes = np.arange(buffer_size)
        self.fn_random_sample,self.fn_log_prob = Normal(next(self.rng),0,1, sample_maxima=False)

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        mean, _ , _ = self.policy.apply(params_policy, state)
        return jnp.clip(mean,-1,1)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        #no randomisation done by actor apply!!
        mean, _ ,log_std = self.policy.apply(params_policy, state)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, True)

    def update(self, wandb_run):
        #get buffer values
        print('getting_instance')
        outputs = self.buffer.get()
        states, actions, rewards, dones, log_pi_olds, next_states = outputs
        # Calculate gamma-retwandb_runurns and GAEs.
        print('Calculate gamma-returns and GAEs.')
        gaes, targets = self.calculate_gae(
            params_policy=self.params_policy,
            actors_states=states,
            actors_rewards=rewards,
            actors_dones=dones,
            actors_next_states=next_states,
        )
        #
        dataframe= (states, actions, rewards, dones, log_pi_olds, next_states,gaes,targets)
        n_outputs= []
        for data in dataframe:
            reshaped = jnp.reshape(data,(-1,)+data.shape[2:])
            n_outputs.append(reshaped)

        #rearranged data
        state, action, reward, done, log_pi_old, next_state,gae,target = n_outputs
        idxes = np.arange(len(state))
        for i_count in range(self.epoch_ppo):
            print(f"epoch {i_count}")
            for start in range(0, len(state), self.batch_size):
                self.learning_step += 1
                idx = idxes[start : start + self.batch_size]
                # get critic loss.
                loss_critic,_= self._loss_critic(
                        self.params_policy,
                        state[idx],
                        target[idx],
                )
                                # Update critic.
                self.opt_state_policy, self.params_policy, loss_critic, _ = optimize(
                    self._loss_critic,
                    self.opt_policy,
                    self.opt_state_policy,
                    self.params_policy,
                    self.max_grad_norm,
                    state=state[idx],
                    target=target[idx],
                )
                # Update policy.
                self.opt_state_policy, self.params_policy, loss_actor, aux = optimize(
                    self._loss_actor,
                    self.opt_policy,
                    self.opt_state_policy,
                    self.params_policy,
                    self.max_grad_norm,
                    state=state[idx],
                    action=action[idx],
                    log_pi_old=log_pi_old[idx],
                    gae=gae[idx],
                    ent_coef=self.ent_coef,
                    entropy_loss=-jnp.mean(-log_pi_old[idx]),
                    vf_coef=self.vf_coef,
                    value_loss=loss_critic
                )
                params = self.params_policy.copy()
                log_std = params['log_std']['constant']
                params.pop('log_std')
                wandb.log({"params-log_std":wandb.Histogram(log_std)})
                for layer in params:
                    w = params[layer]['w']
                    b = params[layer]['b']
                    wandb.log({f"params-{layer}-weights":wandb.Histogram(w)})
                    wandb.log({f"params-{layer}-bias":wandb.Histogram(b)})

            #log the losses
            wandb.log({"loss/critic": np.array(loss_critic)})
            wandb.log({"loss/actor": np.array(loss_actor)})

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
        target: np.ndarray,
    ) -> jnp.ndarray:
        return jnp.square(target - self.policy.apply(params_policy, state)[1]).mean(), None

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        log_pi_old: np.ndarray,
        gae: jnp.ndarray,
        ent_coef: jnp.ndarray,
        entropy_loss: jnp.ndarray,
        vf_coef: jnp.ndarray,
        value_loss: jnp.ndarray
    ) -> jnp.ndarray:
        # Calculate log(\pi) at current policy.
        mean, _ ,log_std = self.policy.apply(params_policy, state)
        #log_pi = evaluate_gaussian_and_tanh_log_prob(mean, log_std, action)
        log_pi = self.fn_log_prob(action,mean=mean,sd=log_std)
        # Calculate importance ratio.
        ratio = jnp.exp(log_pi - log_pi_old)
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
        #loss = loss_actor + ent_coef*entropy_loss + vf_coef*value_loss 
        loss = loss_actor + ent_coef*entropy_loss + vf_coef*value_loss 
        return loss, (log_pi)

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self,
        params_policy: hk.Params,
        actors_states: np.ndarray,
        actors_rewards: np.ndarray,
        actors_dones: np.ndarray,
        actors_next_states: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        actors_gae = []
        actors_targets = []
        for state, reward, done, next_state in zip(actors_states,actors_rewards,actors_dones,actors_next_states): 
            # Current and next value estimates.
            _,value,_ = jax.lax.stop_gradient(self.policy.apply(params_policy, state))
            _,next_value,_ = jax.lax.stop_gradient(self.policy.apply(params_policy, next_state))
            # Calculate TD errors.
            delta = reward + self.gamma * next_value * (1.0 - done) - value
            # Calculate GAE recursively from behind.
            gae = [delta[-1]]
            for t in jnp.arange(len(state)- 2, -1, -1):
                gae.insert(0, delta[t] + self.gamma * self.lambd * (1 - done[t]) * gae[0])
            gae = jnp.array(gae)
            actors_gae.append((gae - gae.mean()) / (gae.std() + 1e-8))
            actors_targets.append(gae + value)
        return jnp.array(actors_gae), jnp.array(actors_targets)
