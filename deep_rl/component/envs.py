#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import re
import gym
# import gym_pygame
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from .atari_wrappers import make_atari, wrap_deepmind, FrameStack
from .wrappers import TimeLimit

from collections import defaultdict

from ..utils.torch_utils import random_seed
from ..utils.misc import mkdir

try:
    import roboschool
except ImportError:
    pass


_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


_game_envs['user_env'] = {
    'TwoLoops',
    'CounterExamplefMax',
    'Corridol',
    'FourRoomsOneGoal',
    'FourRoomsOneGoalOptionsandActions',
    'FourRoomsOneGoalOptionsOnly',
    'FourRoomsOneGoalActionsOnly',
    'FourRoomsFourGoals',
    'FourRoomsSixteenGoals',
    'FourRoomsAllGoals',
    'TwoRooms',
    'TwoRoomsThreeGoals',
    'ChangingGoalFourRooms'
}


def get_env_type(env_id):

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        # env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, env_args, seed, timeout, episode_life=True):
    # TODO: add pygame in
    env_type = get_env_type(env_id)

    if env_type == 'dm':
        import dm_control2gym
        _, domain, task = env_id.split('-')
        env = dm_control2gym.make(domain_name=domain, task_name=task)
        # TODO: add timeout
    elif env_type == 'atari' and "NoFrameskip" in env_id and '-ram' not in env_id:
        env = make_atari(env_id, timeout)
        env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=False, frame_stack=False, scale=False)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            env = TransposeImage(env)
        env = FrameStack(env, 4)
    elif env_type == 'user_env':
        from . import user_envs
        env_class = getattr(user_envs, env_id)
        env = env_class(env_args)
        if timeout is not None:
            env = TimeLimit(env, max_episode_steps=timeout)
    elif env_type == 'gym_pygame':
        from . import user_envs
        env_class = getattr(user_envs, env_id)
        env = env_class()
        if timeout is not None:
            env = TimeLimit(env, max_episode_steps=timeout)
    else:
        # other gym games
        env = gym.make(env_id)
        if timeout is not None:
            env = TimeLimit(env.unwrapped, max_episode_steps=timeout)

    env.seed(seed)

    return env


class Env(object):
    def __init__(self,
                 env_id,
                 env_args,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5)),
                 timeout=None
                 ):
        self.env_id = env_id
        self.timeout = timeout
        if log_dir is not None:
            mkdir(log_dir)
        env = make_env(env_id, env_args, seed, timeout, episode_life)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.observation_space.shape))

        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'
        self.eps_return = 0
        self.eps_length = 0

    def step(self, action):
        if isinstance(self.action_space, Box):
            action = np.clip(action, self.action_space.low, self.action_space.high)

        obs, reward, done, info = self.env.step(action)
        self.eps_length += 1
        self.eps_return += reward
        
        if 'TimeLimit.truncated' not in info:
            info['TimeLimit.truncated'] = False
            
        if done:
            info['episodic_return'] = self.eps_return
            info['episodic_length'] = self.eps_length
            self.eps_length = 0
            obs = self.reset()
        else:
            info['episodic_return'] = None
            info['episodic_length'] = self.eps_length
        return obs, reward, done, info

    def reset(self):
        self.eps_return = 0
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


if __name__ == '__main__':
    task = Env('Hopper-v2')
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
