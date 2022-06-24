from pettingzoo.butterfly import pistonball_v6
import supersuit as ss

def environment_setup(test=True):
    #setting up the testing and live cases
    if test:
        num_envs = 1
        num_cpus = 4
    else:
        num_envs = 8
        num_cpus = 12
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125
    )
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84,y_size=84)
    env = ss.frame_stack_v1(env, stack_size=3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, 
        num_envs, 
        num_cpus=num_cpus, 
        base_class='stable_baselines3'
    )
    return env
def play_enviromnet_setup():
    env = pistonball_v6.env(n_pistons=20)
    env = ss.color_reduction_v0(env,mode='B')
    env = ss.resize_v1(env, x_size=84,y_size=84)
    env = ss.frame_stack_v1(env, 3)
    return env
