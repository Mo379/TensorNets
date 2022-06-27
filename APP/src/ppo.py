from functools import partial
from typing import Tuple
#
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
#
from .base_algorithm import OnPolicyActorCritic
from .util import optimise
from src.agent import *
import wandb


class PPO(OnPolicyActorCritic):
    name = "PPO"
    #
    def __init__(
        self,
        #seed and root
        seed,
        root,
        #models and model hyperparams
        fn_policy,
        lr_policy,
        #algorithm hyper params
        max_grad_norm,
        gamma,
        clip_eps,
        lambd,
        ent_coef,
        vf_coef,
        #env hyperparams
        state_space,
        action_space,
        #training length hyperparams
        num_agent_steps,
        buffer_size,
        batch_size,
        epoch_ppo,
    ):
        #assertion
        assert buffer_size % batch_size == 0
        #super
        super(PPO, self).__init__(
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
        # Policy.
        self.policy= hk.without_apply_rng(hk.transform(fn_policy))
        self.params_policy = self.policy.init(next(self.rng), *self.fake_args_policy)
        #self.load_params('logs/pkls')
        opt_init, self.opt_policy= optax.adam(lr_policy)
        self.opt_state_policy= opt_init(self.params_policy)
        # Other parameters.
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.ent_coef=vf_coef
        self.vf_coef=vf_coef
        self.max_grad_norm = max_grad_norm
        self.idxes = np.arange(buffer_size)

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        mean, _ , _ = self.policy.apply(params_policy, state)
        return mean

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        #no randomisation done by actor apply!!
        mean, _ ,log_std = self.policy.apply(params_policy, state)
        #use distribution 

        return action,log_pi

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
                self.opt_state_policy, self.params_policy, loss_critic, _ = optimise(
                    self._loss_critic,
                    self.opt_policy,
                    self.opt_state_policy,
                    self.params_policy,
                    self.max_grad_norm,
                    state=state[idx],
                    target=target[idx],
                )
                # Update policy.
                self.opt_state_policy, self.params_policy, loss_actor, aux = optimise(
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
            #log the losses
            wandb.log({"loss/critic": np.array(loss_critic)})
            wandb.log({"loss/actor": np.array(loss_actor)})

