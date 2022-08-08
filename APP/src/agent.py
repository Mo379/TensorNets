# system
# ML
import haiku as hk
import jax
import jax.numpy as jnp
from jax.lax import fori_loop


# initialisers
initializer = hk.initializers.VarianceScaling(
        1.0,
        "fan_avg",
        "truncated_normal"
    )
initializer_bias = jnp.zeros


# feature extractor
def feature_extractor(x):
    x = jnp.array(x, dtype=jnp.float32)
    x = x/255.0
    x = hk.Conv2D(
        32, (8, 8),
        stride=(4, 4),
        padding='VALID',
        with_bias=True,
        w_init=initializer,
        b_init=initializer_bias,
        name='NatureCNN_l1'
    )(x)
    x = jax.nn.relu(x)
    x = hk.Conv2D(
        64, (4, 4),
        stride=(2, 2),
        padding='VALID',
        with_bias=True,
        w_init=initializer,
        b_init=initializer_bias,
        name='NatureCNN_l2'
    )(x)
    x = jax.nn.relu(x)
    x = hk.Conv2D(
        64, (3, 3),
        stride=(1, 1),
        padding='VALID',
        with_bias=True,
        w_init=initializer,
        b_init=initializer_bias,
        name='NatureCNN_l3'
    )(x)
    x = jax.nn.relu(x)
    x = hk.Flatten(preserve_dims=1)(x)
    x = hk.Linear(
        512,
        with_bias=True,
        w_init=initializer,
        b_init=initializer_bias,
        name='NatureCNN_l4'
    )(x)
    x = jax.nn.relu(x)
    return x


# Policy network
def policy_net(x):
    policy_out = hk.nets.MLP(
            [1],
            with_bias=True,
            w_init=initializer,
            b_init=initializer_bias,
            name='policy_net'
        )(x)
    return policy_out


# value network
def value_net(x):
    value_out = hk.nets.MLP(
        [1],
        with_bias=True,
        w_init=initializer,
        b_init=initializer_bias,
        name='value_net'
    )(x)
    return value_out


# Tensor Policy network
def tn_policy_net(x):
    x = ActorCriticTensorNet(
            32, 3,
            is_actor=True,
            name='policy_network_1'
        )(x)
    # normilsation and sigmoid activation
    return x


# value network
def tn_value_net(x):
    x = ActorCriticTensorNet(
            32, 1,
            is_actor=False,
            name='value_network_1'
            )(x)
    return x


# defining the tensor network layer
class ActorCriticTensorNet(hk.Module):
    # Init function
    def __init__(
            self,
            contraction_dim_size,
            output_dim_size,
            is_actor=True,
            name=None
    ):
        super().__init__(name=name)
        # save params
        self.contraction_dim_size = contraction_dim_size
        self.output_dim_size = output_dim_size
        self.is_actor = is_actor

    # call function
    def __call__(self, inputs):
        mps_params = hk.get_parameter(
            "mps",
            # dims(agent,feature_vec,contraction,output)
            shape=(
                inputs.shape[0],
                inputs.shape[1],
                self.output_dim_size,
                self.contraction_dim_size,
                self.contraction_dim_size
            ),
            dtype=inputs.dtype,
            init=initializer
        )
        if self.is_actor:
            left_environments = self.calculate_left_environments(
                    mps_params, inputs
                )
            log_prob, (action, key) = self.policy_head(
                    inputs,
                    hk.next_rng_key(),
                    mps_params,
                    left_environments
                )
            return action, (log_prob)
        else:
            value = self.value_function_head(mps_params, inputs)
            return value

    def tensor_scan(self, embedding_vectors, mps_params):
        """performs the contraction of the mps with the embedding vectors.
        Args:
            embedding_vectors (array):
                shape = (# agents, embedding_dim, bond_dim, bond_dim)
            mps_params (array):
                shape = (# agents, embedding_dim, bond_dim, bond_dim)
        Returns:
            float : the value of the traced mps-embedding product.
        """
        env = jnp.tensordot(
                embedding_vectors[0], mps_params[0], axes=((0), (0))
            )
        val = (env, embedding_vectors, mps_params)
        val = fori_loop(1, len(embedding_vectors), self._tensor_step, val)
        return jnp.trace(val[0].T).T

    # A tensornet step function
    def _tensor_step(self, step, val):
        """Contracts the left-environment onto the next matrix and
            embedding vector in the product
        """
        env, embedding_vectors, mps_params = val
        mat = jnp.tensordot(
                embedding_vectors[step], mps_params[step], axes=((0), (0))
                )
        env = jnp.matmul(env, mat)
        env = jax.nn.sigmoid(env)
        return env, embedding_vectors, mps_params

    def value_function_head(self, mps_params, embedding_vectors):
        """Example value function head that returns a value for each agent"""
        value = self.tensor_scan(embedding_vectors, mps_params)
        return value

    def _tensor_step_right_policy(self, carry, inputs):
        """Contracts the left-environment onto the mpo_params corresponding
            to the policy weights with shape:
            (#agents, d_embedding, d_actions, chi_policy, chi_policy)
        """

        weights, mat, environments, embedding_vectors = carry
        site_idx = inputs
        # attached the embedding vector for the state
        tmp = jnp.tensordot(
                embedding_vectors[site_idx], weights[site_idx], axes=((0), (0))
            )
        # sum over actions
        tmp = jnp.tensordot(tmp, tmp, axes=((0), (0)))
        # reorder into matrix ordering
        # (top-left, bottom-left, top-right, bottom-right)
        tmp = jnp.transpose(tmp, (0, 2, 1, 3))
        tmp = jnp.reshape(
                tmp, (tmp.shape[0]*tmp.shape[1], tmp.shape[2]*tmp.shape[3])
            )
        # multiply on the result stored in "mat".
        # This now stores the left-environment including the current site
        mat = jnp.matmul(mat, tmp)
        # This is the left_environment inclusive of the
        # current site for the next site
        environments = environments.at[site_idx+1, :, :].set(mat)

        carry = (weights, mat, environments, embedding_vectors)

        output = None

        return carry, output

    def calculate_left_environments(self, weights, embedding_vectors):
        """ applies the step right across the mpo via a scan
            function to get the left environments
        """
        # Initate the current left-environment as identity.
        # This is for site_idx=0.
        mat = jnp.identity(weights.shape[3]**2)

        environments = jnp.zeros(
            # initiate environments = number of sites + 1.
            # trace of last one is just the norm.
            (
                len(embedding_vectors)+1,
                weights.shape[3]**2,
                weights.shape[3]**2)
            )
        # this is the left_environment for site_idx=0 (just identity)
        environments = environments.at[0, :, :].set(mat)
        # carried through the scan
        carry = (weights, mat, environments, embedding_vectors)
        # inputs are the site indices
        inputs = jnp.array(range(len(embedding_vectors)))
        carry, outputs = jax.lax.scan(
                self._tensor_step_right_policy, carry, inputs
            )
        # Tr[environments[-1]] is the normalisation factor.
        weights, mat, environments, embedding_vectors = carry
        return environments

    def _get_prob_vector(
            self,
            site_idx,
            weights,
            embedding_vectors,
            left_environments,
            right_environment
    ):
        """
        Calculates the probability vector for actions at the site where the sites
        to the left are marginalised (left_environments contains sums over actions)
        and right_environment has conditioning (actions have been selected)
        Args:
            site_idx (_type_): _description_
            weights (_type_): _description_
            left_environments (_type_): _description_
            right_environment (_type_): _description_
        Returns:
            _type_: _description_
        """

        # attached the embedding vector for the state
        tmp = jnp.tensordot(
                embedding_vectors[site_idx], weights[site_idx], axes=((0), (0))
            )
        # split the right environment index
        environment = jnp.reshape(
                left_environments[site_idx],
                (
                    left_environments.shape[1],
                    weights.shape[3],
                    weights.shape[3]
                )
            )
        environment = jnp.tensordot(environment, tmp, axes=((1), (1)))
        tmp = jnp.tensordot(environment, tmp, axes=((1), (1)))
        # split the left-index
        right_environment = jnp.reshape(
                right_environment,
                (
                    weights.shape[3],
                    weights.shape[3],
                    right_environment.shape[1]
                )
            )
        tmp = jnp.tensordot(tmp, right_environment, axes=((2, 4), (0, 1)))
        density_matrix = jnp.trace(tmp, axis1=0, axis2=3)
        probs_unnormed = jnp.diagonal(density_matrix)
        return probs_unnormed

    def _tensor_step_left_policy(self, carry, inputs):
        """
        Steps to the left in a leftward sweep. This samples an action and then
        constructs the next right-environment conditioned on the result
        """

        weights, mat, environments, embedding_vectors, key = carry
        site_idx = inputs

        # sample an action
        probs_unnormed = self._get_prob_vector(
                site_idx, weights, embedding_vectors, environments, mat
            )
        norm = jnp.sum(probs_unnormed)
        probs = probs_unnormed/norm
        key, subkey = jax.random.split(key)
        # choose the action
        action = jax.random.choice(
                subkey, jnp.array(list(range(len(probs)))), p=probs
            )
        prob = probs[action]

        # condition on action for next step
        # select action and attached the embedding vector for the state
        tmp = jnp.tensordot(
                embedding_vectors[site_idx],
                weights[site_idx, :, action, :, :], axes=((0), (0))
            )
        # split the left-index
        right_environment = jnp.reshape(
                mat,
                (weights.shape[3], weights.shape[3], mat.shape[1])
            )
        right_environment = jnp.tensordot(
                tmp, right_environment, axes=((1), (0))
            )
        right_environment = jnp.tensordot(
                tmp, right_environment, axes=((1), (1))
            )
        right_environment = jnp.transpose(right_environment, (0, 1, 2))
        # now includes the present site with conditioned action
        mat = jnp.reshape(right_environment, (mat.shape[0], mat.shape[1]))
        carry = (weights, mat, environments, embedding_vectors, key)
        output = (action, prob, norm, probs_unnormed)

        return carry, output

    def sweep_to_left_policy(
            self, weights, embedding_vectors, left_environments, key
    ):
        """
        Full sweep to the left to sample an action and return its log prob
        """
        # initiate the right environment
        mat = jnp.identity(weights.shape[3]**2)
        # carried through the scan
        carry = (weights, mat, left_environments, embedding_vectors, key)
        # inputs are the site indices from last to 0
        inputs = jnp.array(range(len(embedding_vectors)-1, -1, -1))
        carry, outputs = jax.lax.scan(
                self._tensor_step_left_policy, carry, inputs)
        return carry, outputs

    def policy_head(
            self,
            embedding_vectors,
            key,
            policy_weights,
            left_environments
    ):
        """Takes in embedding vectors and samples an action

        Args:
            embedding_vectors (_type_): _description_
            key (_type_): _description_
            policy_weights (_type_): _description_
        Returns:
            pyTree (tuple): log_probability of the sampled action,
            along with the action and key
        """
        carry_leftwards, outputs_leftwards = self.sweep_to_left_policy(
                policy_weights, embedding_vectors, left_environments, key
            )
        _, _, _, _, key = carry_leftwards
        action = outputs_leftwards[0]
        chain_rule_probs = outputs_leftwards[1]
        log_prob_of_action = jnp.sum(jnp.log(chain_rule_probs))
        return log_prob_of_action, (action, key)

    # Normilisation step
    def _norm_step(self, step, val):
        env, state, mps_params = val
        env = jnp.tensordot(
                env, mps_params[:, state[step], :, :], axis=((0), (1))
            )
        env = jnp.tensordot(
                env, mps_params[:, state[step], :, :], axis=((0, 1), (1, 0))
            )
        return env, state, mps_params


class log_std(hk.Module):
    # init function
    def __init__(self, name=None):
        super().__init__(name=name)

    # Call function
    def __call__(self, actions_means):
        sd = hk.get_parameter(
                "constant",
                shape=(1,),
                dtype=actions_means.dtype,
                init=jnp.zeros
            )
        return actions_means, sd


# actor critic nature cnn model
def my_model(x):
    features = feature_extractor(x)
    #
    values = value_net(features)
    action_mean = policy_net(features)
    #
    actions, log_sd = log_std(name='log_std')(action_mean)
    #
    return actions, values, log_sd


# actor critic tensornet model
def my_model_tensornet(x):
    # get input features
    x = feature_extractor(x)
    x = hk.Linear(
        64,
        with_bias=True,
        w_init=initializer,
        b_init=initializer_bias,
        name='Tensornet_dim_reduction'
    )(x)
    x = jax.nn.log_softmax(x)
    #
    values = tn_value_net(x)
    action_means, log_prob = tn_policy_net(x)
    actions, sd = log_std(name='log_std')(action_means)
    return actions, values.reshape((-1, 1)), sd
