from collections import defaultdict
from rl_experiments.common.utils import wrappers

import gym.wrappers
import gym

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


def get_env_type(env_name: str) -> str:
    env_type = None
    for key, val in _game_envs.items():
        if env_name in _game_envs[key]:
            env_type = key

    return env_type


def make_env(env_name: str, seed=None):

    env_type = get_env_type(env_name)

    if env_type is None:
        raise ValueError('Unknown environment name')

    if env_type == 'classic_control':
        env = gym.make(env_name)
    elif env_type == 'atari':

        # env = gym.make(env_name)
        # env = wrappers.NoopResetEnv(env)
        # env = wrappers.MaxAndSkipEnv(env, skip=4)
        # env = wrappers.EpisodicLifeEnv(env)
        # env = wrappers.FireResetEnv(env)
        # env = wrappers.ClipRewardEnv(env)
        # env = wrappers.WarpFrame(env)
        # env = wrappers.FrameStack(env, 4)
        env = gym.make(env_name)
        env = gym.wrappers.AtariPreprocessing(env, scale_obs=False)
        env = wrappers.EpisodicLifeEnv(env)
        env = wrappers.FireResetEnv(env)
        env = wrappers.ClipRewardEnv(env)
        env = gym.wrappers.FrameStack(env, 4)

    else:
        raise ValueError('Unknown environment type')

    if seed is not None:
        env.seed(seed)

    return env
