import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.lax import cond, fori_loop
from jax.config import config
config.update("jax_enable_x64", True)
from agent import *


#value net generalised
def tensor_scan(embedding_vectors, mps_params):
    env = jnp.tensordot(embedding_vectors[0], mps_params, axes=((0),(0)))
    val = (env, embedding_vectors, mps_params)
    val = fori_loop(1, len(embedding_vectors), _tensor_step, val)
    return jnp.trace(val[0].T).T
def _tensor_step(step, val):
    env, embedding_vectors, mps_params = val
    mat = jnp.tensordot(embedding_vectors[step], mps_params, axes=((0),(0)))
    env = jnp.matmul(env, mat)
    env = jax.nn.sigmoid(env)
    return env, embedding_vectors, mps_params
def _norm_step(step, val):
    env, state, mps_params = val
    env = jnp.tensordot(env, mps_params[:,state[step], :, :], axis=((0),(1)))
    env = jnp.tensordot(env, mps_params[:,state[step], :, :], axis=((0,1),(1,0)))
    return env, state, mps_params

#20,84,84,3 -> feature extractor -> 20,embedding_vector_size -> [tensornetwork] -> log_std -> action,value,sd
key = random.PRNGKey(0)
key2 = random.PRNGKey(2)
n = 20
embedding_vectors = random.normal(key, (n,64))
mps_params = random.normal(key2, (64,n,1,16,16))
print(tensor_scan(embedding_vectors,mps_params).shape)
print(tensor_scan(embedding_vectors,mps_params).mean())

#init model
example_batch= random.normal(key, (20,84,84,3))
model_init,model_apply = hk.without_apply_rng(hk.transform(my_model_tensornet))
model_params = model_init(key,example_batch)
print(jax.tree_map(lambda x: x.shape,model_params))
output = model_apply(model_params,example_batch)
for o in output:
    print(o.shape)

import graphviz
def test_model_visualisation():
    batch_input = jax.random.normal(key,(20,84,84,3))
    dot = hk.experimental.to_dot(model_apply)(
            model_params,batch_input)
    try:
        graphviz.Source(dot).render('output/model_graph')
        status =1
    except:
        status =0

x = test_model_visualisation()
print(x)
#policy net components
def _norm_step(step, val):
    env, state, mps_params = val
    combined_mat = jnp.tensordot(
        mps_params[:,state[step], :, :], 
        mps_params[:,state[step], :, :], axis=((0),(0))
    )
    # returns a 4-index object (l1,r1,l2,r2) shape (CHI,CHI,CHI,CHI)
    combined_mat = jnp.transpose(combined_mat, (0,2,1,3))
    combined_mat = jnp.reshape(combined_mat,(CHI**2,CHI**2))
    env = jnp.matmul(env, combined_mat)
    return env, state, mps_params





















