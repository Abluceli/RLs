from copy import deepcopy
# from gym_wrapper import gym_envs
import sys
sys.path.append("/Users/yangli/GitHub/")
from UnityEnvTools.mlagents.envs.environment import UnityEnvironment
from common.unity_wrapper import InfoWrapper, UnityReturnWrapper, SamplerWrapper, UnityReturnWrapper_GCN


def make_env(env_args):
    if env_args['type'] == 'gym':
        # env = make_gym_env(env_args)
        pass
    elif env_args['type'] == 'unity':
        env = make_unity_env(env_args)
    else:
        raise Exception('Unknown environment type.')
    return env


# def make_gym_env(env_args):
#     env_kargs = deepcopy(env_args)
#     env = gym_envs(env_kargs)
#     return env


def make_unity_env(env_args):
    if env_args['file_path'] is None:
        env = UnityEnvironment()
    else:
        env = UnityEnvironment(
            file_name=env_args['file_path'],
            base_port=env_args['port'],
            no_graphics=not env_args['render']
        )
    env = InfoWrapper(env)
    env = UnityReturnWrapper_GCN(env)
    env = SamplerWrapper(env, env_args)
    return env
