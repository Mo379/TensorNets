from src.util import *
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np

# Initialising the feature extractor
model_features= hk.transform(feature_extractor)
rng = jax.random.PRNGKey(0)
examples = jax.random.normal(rng,(1,84,84,4))
model_features_params = model_features.init(rng, examples)

#
transferred_params = transfer_params(trained_params['policy'], model_features_params)
if __name__ == '__main__':
    print('hello world')







