from src.network_components import *

def my_model(x, deterministic=True):
    features = feature_extractor(x)
    #
    values = value_net(features)
    action_mean = policy_net(features)
    #
    actions, sd= log_std(name='log_std', deterministic=deterministic)(action_mean)
    #
    return actions,values,sd

def my_actor(x, deterministic=True):
    features = feature_extractor(x)
    #
    action_mean = policy_net(features)
    #
    actions, sd = log_std(name='log_std', deterministic=deterministic)(action_mean)
    #
    return actions,sd

def my_critic(x):
    features = feature_extractor(x)
    #
    values = value_net(features)
    #
    #
    return values




def my_TN_model(x):
    features = feature_extractor(x)
    pass
