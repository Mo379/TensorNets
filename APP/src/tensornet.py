# %%
# System
# ML
from agent import my_model_tensornet
import jax
import jax.numpy as jnp
from jax import random
from jax.lax import fori_loop
from jax.config import config
import haiku as hk
import graphviz
config.update("jax_enable_x64", True)


# value net generalised
def tensor_scan(embedding_vectors, mps_params):
    """performs the contraction of the mps with the embedding vectors.

    Args:
        embedding_vectors (array): shape = (# agents, embedding_dim, bond_dim, bond_dim)
        mps_params (array): shape = (# agents, embedding_dim, bond_dim, bond_dim)

    Returns:
        float : the value of the traced mps-embedding product.
    """
    env = jnp.tensordot(embedding_vectors[0], mps_params[0], axes=((0), (0)))
    val = (env, embedding_vectors, mps_params)
    val = fori_loop(1, len(embedding_vectors), _tensor_step, val)
    return jnp.trace(val[0].T).T

# A tensornet step function
def _tensor_step(step, val):
    """Contracts the left-environment onto the next matrix and embedding vector in the product"""
    env, embedding_vectors, mps_params = val
    mat = jnp.tensordot(embedding_vectors[step], mps_params[step], axes=((0), (0)))
    env = jnp.matmul(env, mat)
    env = jax.nn.sigmoid(env)
    return env, embedding_vectors, mps_params

def value_function_head(mps_params, embedding_vectors):
    """Example value function head that returns a value for each agent"""
    value = tensor_scan(embedding_vectors, mps_params)
    return jnp.tile(value, [len(embedding_vectors),1])

# Normilisation step
def _norm_step(step, val):
    env, state, mps_params = val
    env = jnp.tensordot(
            env, mps_params[:, state[step], :, :], axis=((0), (1))
        )
    env = jnp.tensordot(
            env, mps_params[:, state[step], :, :], axis=((0, 1), (1, 0))
        )
    return env, state, mps_params


# Visualise the model
def test_model_visualisation():
    batch_input = jax.random.normal(key, (20, 84, 84, 3))
    dot = hk.experimental.to_dot(model_apply)(
            model_params, batch_input
        )
    try:
        graphviz.Source(dot).render('/workdir/APP/src/output/model_graph')
        result = True
    except Exception:
        result = False
    return result

# %%
# keys
key = random.PRNGKey(0)
key2 = random.PRNGKey(2)
NUM_AGENTS = 20  # number of agents
EMBEDDING_DIM = 64
BOND_DIM = 16
embedding_vectors = random.normal(key, (NUM_AGENTS, EMBEDDING_DIM))
mps_params = random.normal(key2, (NUM_AGENTS, EMBEDDING_DIM, BOND_DIM, BOND_DIM))
print(tensor_scan(embedding_vectors, mps_params).shape)
print(tensor_scan(embedding_vectors, mps_params))
print(tensor_scan(embedding_vectors, mps_params).mean())
print(value_function_head(mps_params, embedding_vectors))
# init model
example_batch = random.normal(key, (20, 84, 84, 3))
model_init, model_apply = hk.without_apply_rng(
        hk.transform(my_model_tensornet)
    )
model_params = model_init(key, example_batch)
print(jax.tree_map(lambda x: x.shape, model_params))
output = model_apply(model_params, example_batch)
for o in output:
    print(o.shape)
x = test_model_visualisation()
print(x)
# test normilisation


















# %%
