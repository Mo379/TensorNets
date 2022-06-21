import jax.numpy as jnp

def transfer_params(trained_params, model_params):
  transferred_dict = {}
  #log_std layer
  trained_log_std = 'log_std'
  log_std_constant = trained_params[trained_log_std].numpy()
  transferred_dict.update({ 'log_std': {
          'constant': jnp.round(log_std_constant,4)
      } 
  })

  #reset of the network
  trained_params.pop('log_std')
  trained_keys =  list(trained_params.keys())
  trained_keys = [trained_keys[i:i+2] for i in range(0, len(trained_keys), 2)]
  model_params.pop('log_std')
  model_keys = list(model_params.keys())
  desired_order_list = [0,1, 2, 3, 5, 4]
  model_keys = [model_keys[k] for k in desired_order_list]
  for i,keys in enumerate(zip(trained_keys,model_keys)):
    trained_keys, my_keys = keys
    #get layer params
    weights = trained_params[trained_keys[0]].numpy()
    bias = trained_params[trained_keys[1]].numpy()
    if i in [0,1,2]:
        weights = jnp.flip(weights.T)
        bias = jnp.flip(bias)
    if i in [3,4,5]:
        weights = jnp.flip(weights.T)
        bias = jnp.flip(bias)
    transferred_dict.update({
      my_keys: {
          'w': jnp.round(weights, 4),
          'b': jnp.round(bias,4)
      } 
    })
  return transferred_dict











