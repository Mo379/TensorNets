import os
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
from src.util import *
import supersuit as ss
import haiku as hk
import jax
import jax.numpy as jnp 
import numpy as np
import graphviz
from pathlib import Path
import torch
import torchvision as tv
import unittest

class TestModels(unittest.TestCase):
    def setUp(self):
        self.model_features= hk.transform(my_model)
        self.rng = jax.random.PRNGKey(0)
        self.examples = jax.random.normal(self.rng,(1,84,84,3))
        self.model_features_params = self.model_features.init(self.rng, self.examples)
    #
    def test_model_apply_ouput(self):
        output = self.model_features.apply(self.model_features_params,self.rng,self.examples)
        self.assertEqual(type(output),tuple)
        self.assertEqual(output[0].shape,(1,1))
        self.assertEqual(output[1].shape,(1,1))
        self.assertEqual(output[2].shape,(1,1))
    def test_model_structure(self):
        layer_names = [
                'log_std',
                'NatureCNN_l1',
                'NatureCNN_l2',
                'NatureCNN_l3',
                'flatten',
                'NatureCNN_l4',
                'policy_net',
                'policy_net/~/linear_0',
                'value_net',
                'value_net/~/linear_0',
                ]
        names = []
        for i in hk.experimental.eval_summary(self.model_features)(self.examples):
            #mod := NatureCNN_l1   | in := f32[1,84,84,4] out := f32[1,20,20,32]
            names.append(i.module_details.module.module_name)
            input_shape = i.args_spec[0]
            output_shape = i.output_spec
        self.assertEqual(len(set(names) & set(layer_names)), len(names))
    def test_model_visualisation(self):
        batch_input = jax.random.normal(self.rng,(32,84,84,3))
        dot = hk.experimental.to_dot(self.model_features.apply)(
                self.model_features_params,self.rng,batch_input)
        try:
            graphviz.Source(dot).render('output/model_graph')
            status =1
        except:
            status =0
        self.assertEqual(status,1)
    def test_PPO_model_output(self):
        env = environment_setup()
        model = PPO(
            CnnPolicy,env,verbose=3,gamma=0.95, 
            n_steps=16,ent_coef=0.0905168,learning_rate=0.00062211, 
            vf_coef=0.042202,max_grad_norm=0.9,gae_lambda=0.99, 
            n_epochs=2,clip_range=0.3,batch_size=16
        )
        root= Path(__file__).resolve().parent.parent
        _path = os.path.join(root,'pkls/models/policy_test')
        model.learn(total_timesteps=5000)
        try: 
            model.save(_path)
        except:
            status = 0
        else:
            status=1
        # load environemnt examples
        env_examples = jax.random.normal(self.rng,(32,84,84,3))
        tensor_examples= np.array(env_examples)
        #
        root= Path(__file__).resolve().parent.parent
        _path = os.path.join(root,'pkls/models/policy_test')
        model = PPO.load(_path)
        model_predictions= model.predict(tensor_examples)
        #
        self.assertEqual(type(model_predictions[0]),np.ndarray)
        self.assertEqual(type(model_predictions[1]),type(None))
        self.assertEqual(status,1)
    def test_model_comparison(self):
        root= Path(__file__).resolve().parent.parent
        _path = os.path.join(root,'pkls/models/policy_test')
        model = PPO.load(_path)
        trained_params = model.get_parameters()
        #
        pytorch_model= jax.tree_map(
            lambda x: x.shape, 
            trained_params['policy']
        )
        haiku_model = jax.tree_map(
            lambda x: x.shape, 
            self.model_features_params
        )
        self.assertEqual(len(pytorch_model),len(haiku_model)*2 - 1)

    def test_parameter_transfer_functionality(self):
        root= Path(__file__).resolve().parent.parent
        _path = os.path.join(root,'pkls/models/policy_test')
        model = PPO.load(_path)
        trained_params = model.get_parameters()
        #
        example_params = self.model_features_params.copy()
        transferred_params = transfer_params(trained_params['policy'], example_params)
        transferred_shapes = jax.tree_map(lambda x,y: x.shape, transferred_params,self.model_features_params)
        transferred_forward = self.model_features.apply(transferred_params,self.rng,self.examples)
        self.assertEqual(type(transferred_forward[0]),type(jnp.array([1])))
        self.assertEqual(type(transferred_forward[1]),type(jnp.array([1])))
        self.assertEqual(transferred_forward[0].shape,(1,1))
        self.assertEqual(transferred_forward[1].shape,(1,1))
    def test_transfer_trained_output_comparison(self):
        root= Path(__file__).resolve().parent.parent
        _path = os.path.join(root,'pkls/models/model')
        model = PPO.load(_path)
        trained_params = model.get_parameters()
        #execute parameter transfer
        transferred_params = transfer_params(trained_params['policy'], self.model_features_params)
        # load environemnt examples
        env_examples = jax.random.normal(self.rng,(32,84,84,3))
        tensor_examples= np.array(env_examples)
        # get both model predictions
        model_predictions= model.predict(tensor_examples, deterministic=False)
        transferred_predictions,_,_= self.model_features.apply(transferred_params,self.rng,env_examples)
        #
        print(model_predictions, '\n \n')
        print(transferred_predictions, '\n \n')
        self.assertEqual(model_predictions[0][0], transferred_predictions[0][0])

if __name__ == '__main__':
    unittest.main()





