from functools import partial
from typing import Tuple
#
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import distrax
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
        #assert buffer_size % batch_size == 0
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
        #optax set to zero
        self.optax_zero_init,self.optax_zero_apply = optax.set_to_zero()

    @partial(jax.jit, static_argnums=0)
    def _get_value(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        _, value , _ = self.policy.apply(params_policy, state, deterministic=True)
        return value
    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        action, _ , log_std = self.policy.apply(params_policy, state, deterministic=True)
        dist = distrax.MultivariateNormalDiag(action,jnp.ones_like(action)*log_std)
        action = jnp.clip(action,-1,1)
        #
        log_prob = dist.log_prob(action)
        return action, log_prob

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        #randomisation done by policy apply!!
        action, _ ,sd= self.policy.apply(params_policy, state, deterministic=False)
        #
        dist = distrax.MultivariateNormalDiag(action,jnp.ones_like(action)*sd)
        #
        actions= dist.sample(seed=key) 
        actions = jnp.clip(actions,-1,1)
        #
        log_prob = dist.log_prob(actions)
        return actions,log_prob

    def update(self, wandb_run):
        #get buffer items and calculate state value
        print('getting_instance')
        outputs = self.buffer.get()
        actor_first_output= []
        for output in outputs:
            actor_first_output.append(jnp.swapaxes(output, 0,1))
        A_states, A_actions, A_rewards, A_dones, A_log_pi_olds, A_next_states = actor_first_output
        #calculte multi agent state value
        A_values_t= []
        A_rewards_t= []
        A_values_T= []
        for state,reward,next_state in zip(A_states,A_rewards,A_next_states): 
            A_values_t.append(self._get_value(self.params_policy,state))
            A_values_T.append(self._get_value(self.params_policy,next_state))
            A_rewards_t.append(reward)
        A_values_t = jnp.array([A_values_t]).squeeze()
        A_rewards_t = jnp.array([A_rewards_t]).squeeze()
        A_discounts_t = jnp.array([A_values_T]).squeeze()*self.gamma
        # for each agent: Calculate gamma-retwandb_runurns and GAEs.
        print('Calculate gamma-returns and GAEs.')
        A_gae_agents = []
        for rewards,discounts,values in zip(A_rewards_t,A_discounts_t,A_values_t):
            gaes = rlax.truncated_generalized_advantage_estimation(
                    rewards[1:], 
                    discounts[1:], 
                    self.lambd, 
                    values, 
                    True
                )
            #normalising Gaes
            gaes= (gaes- gaes.mean()) / (gaes.std() + 1e-8)
            A_gae_agents.append(gaes)
        A_gae_agents = jnp.array(A_gae_agents)
        A_targets = A_gae_agents + A_values_t[:][:,:-1]
        #
        A_dataframe= (A_states[:][:,:-1], A_actions[:][:,:-1], A_rewards_t[:][:,:-1], A_dones[:][:,:-1], \
                A_log_pi_olds[:][:,:-1], A_next_states[:][:,:-1],A_gae_agents,A_targets)
        A_shuffler = np.random.permutation(A_states.shape[0]*A_states.shape[1])
        A_n_outputs= []
        for data in A_dataframe:
            reshaped = jnp.reshape(data,(-1,)+data.shape[2:])
            reshaped = reshaped[A_shuffler]
            A_n_outputs.append(reshaped)
        #rearranged data
        B_state, B_action, B_reward, B_done, B_log_pi_old, B_next_state,B_gae,B_target = A_n_outputs
        idxes = np.arange(len(B_state))
        #main loop
        for i_count in range(self.epoch_ppo):
            print(f"epoch {i_count}")
            for start in range(0, len(B_state), self.batch_size):
                #setup
                self.learning_step += 1
                idx = idxes[start : start + self.batch_size]
                #zero updates
                self.opt_state_policy,_= self.optax_zero_apply(self.opt_state_policy,None)
                # Update critic.
                self.opt_state_policy, self.params_policy, loss_critic, aux = optimise(
                    self._loss_critic,
                    self.opt_policy,
                    self.opt_state_policy,
                    self.params_policy,
                    self.max_grad_norm,
                    state=B_state[idx],
                    target=B_target[idx],
                )
                # Update policy.
                self.opt_state_policy, self.params_policy, loss_actor, aux = optimise(
                    self._loss_actor,
                    self.opt_policy,
                    self.opt_state_policy,
                    self.params_policy,
                    self.max_grad_norm,
                    state=B_state[idx],
                    action=B_action[idx],
                    log_prob_old=B_log_pi_old[idx],
                    gae=B_gae[idx],
                    ent_coef=self.ent_coef,
                    entropy_loss=-jnp.mean(-B_log_pi_old[idx]),
                    vf_coef=self.vf_coef,
                    value_loss=loss_critic,
                    rng=next(self.rng),
                )
        #log the losses
        wandb.log({"loss/critic": np.array(loss_critic)})
        wandb.log({"loss/actor": np.array(loss_actor)})
        #clear buffer
        self.buffer.clear()
    #loss functions
    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        target: np.ndarray,
    ) -> jnp.ndarray:
        #the input is double batched thus
        MSE = jnp.square(target - self._get_value(params_critic,state)).mean()
        return MSE , None

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        log_prob_old: np.ndarray,
        gae: jnp.ndarray,
        ent_coef: jnp.ndarray,
        entropy_loss: jnp.ndarray,
        vf_coef: jnp.ndarray,
        value_loss: jnp.ndarray,
        rng:jnp.ndarray,
    ) -> jnp.ndarray:
        # Calculate log(\pi) at current policy.
        actions, log_prob= self._explore(params_policy,state,rng)
        # Calculate importance ratio.
        ratio = jnp.exp(log_prob- log_prob_old)
        loss_actor = rlax.clipped_surrogate_pg_loss(ratio, gae, self.clip_eps, use_stop_gradient=True)
        #total loss
        loss = loss_actor + ent_coef*entropy_loss + vf_coef*value_loss 
        return loss, (log_prob)
