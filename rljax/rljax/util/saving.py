import os

import haiku as hk
import numpy as np
import pickle


def save_params(params, path):
    """
    Save parameters.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(params, f)


def load_params(path):
    """
    Load parameters.
    """
    file = open(path,'rb')
    params = pickle.load(file)
    return params
