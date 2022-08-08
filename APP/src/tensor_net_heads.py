# %%
# add introduction,rename to deep network in jax,why jax whats the advantage,
# flexible, deepmind advantages, model structrue

# System
# ML
import jax
import jax.numpy as jnp
from jax import random
from jax.lax import fori_loop, scan
from jax.config import config
import haiku as hk
import graphviz
from jax import value_and_grad
from jax.example_libraries import stax
from jax.example_libraries.stax import\
        (BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax)
config.update("jax_enable_x64", True)


# value net generalised
def tensor_scan(embedding_vectors, mps_params):
    """performs the contraction of the mps with the embedding vectors.

    Args:
        embedding_vectors (array): shape =
            (# agents, embedding_dim, bond_dim, bond_dim)
        mps_params (array): shape =
            (# agents, embedding_dim, bond_dim, bond_dim)

    Returns:
        float : the value of the traced mps-embedding product.
    """
    env = jnp.tensordot(embedding_vectors[0], mps_params[0], axes=((0), (0)))
    val = (env, embedding_vectors, mps_params)
    val = fori_loop(1, len(embedding_vectors), _tensor_step, val)
    return jnp.trace(val[0].T).T


# A tensornet step function
def _tensor_step(step, val):
    """
    Contracts the left-environment onto the next matrix and
    embedding vector in the product
    """
    env, embedding_vectors, mps_params = val
    mat = jnp.tensordot(
            embedding_vectors[step], mps_params[step], axes=((0), (0))
        )
    env = jnp.matmul(env, mat)
    env = jax.nn.sigmoid(env)
    return env, embedding_vectors, mps_params


def value_function_head(embedding_vectors, mps_params):
    """Example value function head that returns a value for each agent"""
    value = tensor_scan(embedding_vectors, mps_params)
    return value


def tile_value(value, number_of_agents):
    return jnp.tile(value, [number_of_agents, 1])


def _tensor_step_right_policy(carry, input):
    """
    Contracts the left-environment onto the mpo_params corresponding
    to the policy weights with shape=
        (#agents, d_embedding, d_actions, chi_policy, chi_policy)
    """

    weights, mat, environments, embedding_vectors = carry
    site_idx = input

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

    # multiply on the result stored in "mat". This now stores the
    # left-environment including the current site
    mat = jnp.matmul(mat, tmp)
    # this is the left_environment inclusive of
    # the current site for the next site
    environments = environments.at[site_idx+1, :, :].set(mat)

    carry = (weights, mat, environments, embedding_vectors)

    output = None

    return carry, output


def calculate_left_environments(weights, embedding_vectors):
    """
    applies the step right across the mpo via a scan
    function to get the left environments
    """
    # Initate the current left-environment as identity. This is for site_idx=0.
    mat = jnp.identity(weights.shape[3]**2)
    # initiate environments = number of sites + 1.
    # trace of last one is just the norm.
    environments = jnp.zeros(
            (
                len(embedding_vectors)+1,
                weights.shape[3]**2,
                weights.shape[3]**2
            )
        )
    # this is the left_environment for site_idx=0 (just identity)
    environments = environments.at[0, :, :].set(mat)

    # carried through the scan
    carry = (weights, mat, environments, embedding_vectors)
    # inputs are the site indices
    inputs = jnp.array(range(len(embedding_vectors)))
    carry, outputs = scan(_tensor_step_right_policy, carry, inputs)
    # Tr[environments[-1]] is the normalisation factor.
    weights, mat, environments, embedding_vectors = carry
    return environments


def _get_prob_vector(site_idx, weights, left_environments, right_environment):
    """
    Calculates the probability vector for actions at the site where
    the sites to the left are marginalised
    (left_environments contains sums over actions) and
    right_environment has conditioning (actions have been selected)

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
            (left_environments.shape[1], weights.shape[3], weights.shape[3])
        )
    environment = jnp.tensordot(environment, tmp, axes=((1), (1)))
    tmp = jnp.tensordot(environment, tmp, axes=((1), (1)))
    # split the left-index
    right_environment = jnp.reshape(
            right_environment,
            (weights.shape[3], weights.shape[3], right_environment.shape[1])
        )
    tmp = jnp.tensordot(tmp, right_environment, axes=((2, 4), (0, 1)))
    density_matrix = jnp.trace(tmp, axis1=0, axis2=3)
    probs_unnormed = jnp.diagonal(density_matrix)

    return probs_unnormed


def _tensor_step_left_policy(carry, input):
    """
    Steps to the left in a leftward sweep. This samples an action and then
    constructs the next right-environment conditioned on the result
    """

    weights, mat, environments, embedding_vectors, key = carry
    site_idx = input

    # sample an action
    probs_unnormed = _get_prob_vector(site_idx, weights, environments, mat)
    norm = jnp.sum(probs_unnormed)
    probs = probs_unnormed/norm
    key, subkey = random.split(key)
    # choose the action
    action = random.choice(subkey, jnp.array(list(range(len(probs)))), p=probs)
    prob = probs[action]

    # condition on action for next step
    # select action and attached the embedding vector for the state
    tmp = jnp.tensordot(
            embedding_vectors[site_idx],
            weights[site_idx, :, action, :, :],
            axes=((0), (0))
        )
    # split the left-index
    right_environment = jnp.reshape(
            mat, (weights.shape[3], weights.shape[3], mat.shape[1])
        )
    right_environment = jnp.tensordot(tmp, right_environment, axes=((1), (0)))
    right_environment = jnp.tensordot(tmp, right_environment, axes=((1), (1)))
    right_environment = jnp.transpose(right_environment, (0, 1, 2))
    # now includes the present site with conditioned action
    mat = jnp.reshape(right_environment, (mat.shape[0], mat.shape[1]))

    carry = (weights, mat, environments, embedding_vectors, key)
    output = (action, prob, norm, probs_unnormed)

    return carry, output


def sweep_to_left_policy(weights, embedding_vectors, left_environments, key):
    """Full sweep to the left to sample an action and return its log prob"""

    # initiate the right environment
    mat = jnp.identity(weights.shape[3]**2)
    # carried through the scan
    carry = (weights, mat, left_environments, embedding_vectors, key)
    # inputs are the site indices from last to 0
    inputs = jnp.array(range(len(embedding_vectors)-1, -1, -1))
    carry, outputs = scan(_tensor_step_left_policy, carry, inputs)
    return carry, outputs


def policy_head(embedding_vectors, key, policy_weights):
    """Takes in embedding vectors and samples an action

    Args:
        embedding_vectors (_type_): _description_
        key (_type_): _description_
        policy_weights (_type_): _description_

    Returns:
        pyTree (tuple): log_probability of the sampled action,
        along with the action and key
    """
    carry_leftwards, outputs_leftwards = sweep_to_left_policy(
            policy_weights, embedding_vectors, left_environments, key
        )
    _, _, _, _, key = carry_leftwards
    action = outputs_leftwards[0]
    chain_rule_probs = outputs_leftwards[1]
    log_prob_of_action = jnp.sum(jnp.log(chain_rule_probs))
    return log_prob_of_action, (action, key)


# keys
key = random.PRNGKey(0)
key2 = random.PRNGKey(2)
NUM_AGENTS = 20  # number of agents
NUM_ACTIONS = 4
EMBEDDING_DIM = 64
BOND_DIM = 16
embedding_vectors = random.normal(key, (NUM_AGENTS, EMBEDDING_DIM))
mps_params = random.normal(
        key2, (NUM_AGENTS, EMBEDDING_DIM, BOND_DIM, BOND_DIM)
    )
print(embedding_vectors.shape)
print(tensor_scan(embedding_vectors, mps_params).shape)
print(tensor_scan(embedding_vectors, mps_params))
print(tensor_scan(embedding_vectors, mps_params).mean())
print('value', value_function_head(embedding_vectors, mps_params))
print(jax.tree_map(lambda x: x.shape, mps_params))
print(len(embedding_vectors))
# init model
# %%
key, subkey = random.split(key)
policy_weights = random.normal(
        subkey, (NUM_AGENTS, EMBEDDING_DIM, NUM_ACTIONS, BOND_DIM, BOND_DIM)
    )

left_environments = calculate_left_environments(
        policy_weights, embedding_vectors
    )
norm = jnp.trace(left_environments[-1])

# %%

SITE_IDX = 19
# Initate the current left-environment as identity. Not stored.
right_environment = jnp.identity(policy_weights.shape[3]**2)

probs_unnormed = _get_prob_vector(
        SITE_IDX, policy_weights, left_environments, right_environment
    )

log_prob, (action, key) = policy_head(embedding_vectors, key, policy_weights)
# %%
policy_grad = value_and_grad(policy_head, argnums=2, has_aux=True)
value_grad = value_and_grad(value_function_head, argnums=1)
# %%
(log_prob, (action, key)), pol_grad = policy_grad(
        embedding_vectors, key, policy_weights
    )
value, val_grad = value_grad(embedding_vectors, mps_params)
# %%
num_classes = 64
batch_size = 1

key = random.PRNGKey(0)
init_fun, conv_net = stax.serial(Conv(32, (5, 5), (2, 2), padding="SAME"),
                                 BatchNorm(), Relu,
                                 Conv(32, (5, 5), (2, 2), padding="SAME"),
                                 BatchNorm(), Relu,
                                 Conv(10, (3, 3), (2, 2), padding="SAME"),
                                 BatchNorm(), Relu,
                                 Conv(10, (3, 3), (2, 2), padding="SAME"),
                                 Relu, Flatten,
                                 Dense(num_classes),
                                 LogSoftmax)
_, params = init_fun(key, (batch_size, 1, 84, 84))


# %%
def value_all(observation, weights):
    cnn_params, mps_params = weights
    cnn_output = conv_net(cnn_params, observation)
    value_output = value_function_head(cnn_output, mps_params)
    return value_output


# %%
def policy_all(observation, key, weights):
    cnn_params, mpo_params = weights
    cnn_output = conv_net(cnn_params, observation)
    log_prob, (action, key) = policy_head(cnn_output, key, mpo_params)
    return log_prob, (action, key)


# %%
observation = random.normal(key, (NUM_AGENTS, 1, 84, 84))
value = value_all(observation, [params, mps_params])
# %%
value_grad = value_and_grad(value_all, argnums=1)
value, val_grad = value_grad(observation, [params, mps_params])
# %%
key, subkey = random.split(key)
log_prob, (action, key) = policy_all(
        observation, key, [params, policy_weights]
    )
policy_grad = value_and_grad(policy_all, argnums=2, has_aux=True)
(log_prob, (action, key)), pol_grad = policy_grad(
        observation, key, [params, policy_weights]
    )
print(action)
# %%
