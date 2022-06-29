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
        states, actions, rewards, dones, log_pi_olds, next_states = actor_first_output
        #calculte multi agent state value
        values_t= []
        rewards_t= []
        values_T= []
        for state,reward,next_state in zip(states,rewards,next_states): 
            values_t.append(self._get_value(self.params_policy,state))
            values_T.append(self._get_value(self.params_policy,next_state))
            rewards_t.append(reward)
        values_t = jnp.array([values_t]).squeeze()
        rewards_t = jnp.array([rewards_t]).squeeze()
        discounts_t = jnp.array([values_T]).squeeze()*self.gamma
        # for each agent: Calculate gamma-retwandb_runurns and GAEs.
        print('Calculate gamma-returns and GAEs.')
        gae_agents = []
        for rewards,discounts,values in zip(rewards_t,discounts_t,values_t):
            gaes = rlax.truncated_generalized_advantage_estimation(
                    rewards[1:], 
                    discounts[1:], 
                    self.lambd, 
                    values, 
                    True
                )
            gaes = jnp.insert(gaes,0,0)
            gae_agents.append(gaes)
        gae_agents = jnp.array(gae_agents)
        targets = gae_agents + values_t
        #
        dataframe= (states, actions, rewards_t, dones, log_pi_olds, next_states,gae_agents,targets)
        shuffler = np.random.permutation(states.shape[0]*states.shape[1])
        n_outputs= []
        for data in dataframe:
            reshaped = jnp.reshape(data,(-1,)+data.shape[2:])
            reshaped = reshaped[shuffler]
            n_outputs.append(reshaped)
        #rearranged data
        state, action, reward, done, log_pi_old, next_state,gae,target = n_outputs
        idxes = np.arange(len(state))
        #main loop
        for i_count in range(self.epoch_ppo):
            print(f"epoch {i_count}")
            for start in range(0, len(state), self.batch_size):
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
                    log_prob_old=log_pi_old[idx],
                    gae=gae[idx],
                    ent_coef=self.ent_coef,
                    entropy_loss=-jnp.mean(-log_pi_old[idx]),
                    vf_coef=self.vf_coef,
                    value_loss=loss_critic,
                    rng=next(self.rng),
                )
            #log the losses
            wandb.log({"loss/critic": np.array(loss_critic)})
            wandb.log({"loss/actor": np.array(loss_actor)})
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
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
        #loss = loss_actor + ent_coef*entropy_loss + vf_coef*value_loss 
        loss = loss_actor + ent_coef*entropy_loss + vf_coef*value_loss 
        return loss, (log_prob)
