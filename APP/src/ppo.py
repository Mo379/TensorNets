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
from .util import optimise, RolloutBuffer
from src.agent import *
import wandb


class PPO():
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
        #seed and root
        self.seed = seed
        #rng
        np.random.seed(self.seed)
        self.rng = PRNGSequence(self.seed)
        self.root = root
        # Policy.
        self.policy= hk.without_apply_rng(hk.transform(fn_policy))
        self.params_policy = self.policy.init(next(self.rng), np.random.normal(size = (1,84,84,3)))
        # Other parameters.
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.ent_coef=vf_coef
        self.vf_coef=vf_coef
        #state and action spaces 
        self.state_space = state_space
        self.action_space = action_space
        #init buffer 
        self.num_agent_steps = num_agent_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epoch_ppo = epoch_ppo



        #optimiser
        opt_init, self.opt_policy= optax.adam(lr_policy)
        self.opt_state_policy= opt_init(self.params_policy)
        #step counts
        self.idxes = np.arange(buffer_size)
        self.buffer = RolloutBuffer()
        self.optax_zero_init,self.optax_zero_apply = optax.set_to_zero()
    #
    def step(self, env, state, done):
        #
        #rng = next()
        #action, log_prob = self._explore(self.params_policy,state,next(self.rng))
        if done.any():
            state = env.reset()
        action, log_prob = self._select_action(self.params_policy,state)
        next_state, reward, done, _ = env.step(action)
        #
        self.buffer.append(state, action, log_prob , reward, done, next_state)
        #
        return next_state, done
    #
    def update(self, wandb_run):
        #get buffer items and calculate state value
        print('getting_instance')
        #get all data 
        outputs = self.buffer.get()
        #
        actor_first_output= []
        for output in outputs:
            output = jnp.array(output)
            actor_first_output.append(jnp.swapaxes(output, 0,1))
        #unpack data
        A_states, A_actions, A_log_pi_olds ,A_rewards, A_dones, A_next_states = actor_first_output
        #getting GAEs and targets
        A_gaes = []
        A_targets = []
        print('Calculating gaes')
        for agent in range(len(A_states)):
            print(f"for agent {agent}")
            gaes, targets = self.calculate_gaes_and_targets_agent(
                    self.params_policy,
                    A_states[agent],
                    A_rewards[agent],
                    A_dones[agent]
                )
            
           # gaes,targets = self.calculate_gae(
           #     self.params_policy,
           #     A_states[agent],
           #     A_rewards[agent],
           #     A_dones[agent],
           #     A_next_states[agent]
           # ) 
            A_gaes.append(gaes)
            A_targets.append(targets)
        A_gaes = jnp.array(A_gaes)
        A_targets = jnp.array(A_targets)
        #align data
        A_dataframe= (A_states, A_actions, A_log_pi_olds,A_rewards, A_dones, \
                     A_next_states,A_gaes,A_targets)
        A_shuffler = np.random.permutation(A_states.shape[0]*A_states.shape[1])
        #flatten data fuse first two dimentions (agent and play-timesetp)
        A_n_outputs= []
        for data in A_dataframe:
            reshaped = jnp.reshape(data,(-1,)+data.shape[2:])
            reshaped = reshaped[A_shuffler]
            A_n_outputs.append(reshaped)
        B_state, B_action, B_log_pi_old ,B_reward, B_done, B_next_state,B_gae,B_target = A_n_outputs
        idxes = np.arange(len(B_state))
        for i_count in range(self.epoch_ppo):
            print(f"epoch {i_count}")
            #
            for start in range(0, len(B_state), self.batch_size):
                #create batches
                idx = idxes[start : start + self.batch_size]
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
                #zero updates
                self.opt_state_policy,_= self.optax_zero_apply(self.opt_state_policy,None)
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
                    vf_coef=self.vf_coef,
                    value_loss=loss_critic,
                    rng=next(self.rng),
                )
        if wandb_run:
            #log the losses
            wandb.log({"loss/critic": np.array(loss_critic)})
            wandb.log({"loss/actor": np.array(loss_actor)})
        self.buffer.clear()








    #proxy model invocations (only use this outside of this class)
    def explore(self, state):
        action, log_prob = self._explore(self.params_policy, state, next(self.rng))
        return action, log_prob
    def select_action(self, state):
        action,log_prob = self._select_action(self.params_policy, state)
        return np.array(action), np.array(log_prob)








    #Model invocations
    @partial(jax.jit, static_argnums=0)
    def _get_value(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        _, value , _ = self.policy.apply(params_policy, state)
        return value
    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action, _ , log_std = self.policy.apply(params_policy, state)
        dist = distrax.MultivariateNormalDiag(action,jnp.ones_like(action)*jnp.exp(log_std))
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
        action, _ ,sd= self.policy.apply(params_policy, state)
        #
        dist = distrax.MultivariateNormalDiag(action,jnp.ones_like(action)*jnp.exp(sd))
        #
        actions= dist.sample(seed=key) 
        actions = jnp.clip(actions,-1,1)
        #
        log_prob = dist.log_prob(actions)
        return actions,log_prob
    #
    @partial(jax.jit, static_argnums=0)
    def _get_entropy(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        action, _ , log_std = self.policy.apply(params_policy, state)
        dist = distrax.MultivariateNormalDiag(action,jnp.ones_like(action)*jnp.exp(log_std))
        return dist.entropy()












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
    #loss actor
    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_policy: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        log_prob_old: np.ndarray,
        gae: jnp.ndarray,
        ent_coef: jnp.ndarray,
        vf_coef: jnp.ndarray,
        value_loss: jnp.ndarray,
        rng:jnp.ndarray,
    ) -> jnp.ndarray:
        # Calculate log(\pi) at current policy.
        actions, log_prob= self._explore(params_policy,state,rng)
        #entropy_loss = self._get_entropy(params_policy,state)
        entropy_loss = 0
        # Calculate importance ratio.
        ratio = jnp.exp(log_prob- log_prob_old)
        loss_actor = rlax.clipped_surrogate_pg_loss(ratio, gae, self.clip_eps, use_stop_gradient=True)
        #total loss
        loss = loss_actor + ent_coef*entropy_loss + vf_coef*value_loss 
        return loss, (log_prob)









    #save and load params 
    def fn_save_params(self,path):
        """
        Save parameters.
        """
        path = os.path.join(path, "params_policy.pkl")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.params_policy, f)
    #
    def fn_load_params(self,path):
        """
        Load parameters.
        """
        path = os.path.join(path, "haiku_transfer.pkl")
        file = open(path, 'rb')
        params = pickle.load(file)
        return params





    @partial(jax.jit, static_argnums=0)
    def calculate_gaes_and_targets_agent(self,params,states,rewards,dones):
        values = self._get_value(params,states).squeeze()
        #
        gaes = []
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
                a_t += discount*(rewards[k] + self.gamma*values[k+1]*\
                        (1-jnp.array(dones[k],int)) - values[k])
                discount *= self.gamma*self.lambd
            gaes.append(a_t)
        gaes.append(0)
        gaes = jnp.array(gaes, jnp.float32)
        targets = gaes*values
        #gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return gaes,targets
    #
    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self,
        params: hk.Params,
        state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Current and next value estimates.
        value = jax.lax.stop_gradient(self._get_value(params, state).squeeze())
        next_value = jax.lax.stop_gradient(self._get_value(params, next_state).squeeze())
        # Calculate TD errors.
        delta = reward + self.gamma * next_value * (1.0 - done) - value
        # Calculate GAE recursively from behind.
        gae = [delta[-1]]
        for t in jnp.arange(self.buffer_size - 2, -1, -1):
            gae.insert(0, delta[t] + self.gamma * self.lambd * (1 - done[t]) * gae[0])
        gae = jnp.array(gae)
        return (gae - gae.mean()) / (gae.std() + 1e-8), gae + value
