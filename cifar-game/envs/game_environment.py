from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


def create_game_env(env_cfg, data):
    return MultiStepGame(env_cfg.seed, env_cfg.share_reward, data,
                         env_cfg.discrete_comm, env_cfg.comm_size)


class Game(gym.Env):
    def __init__(self, seed, share_reward, data, debug=False):
        super(Game, self).__init__()
        self.debug = debug
        self.seed(seed=seed)
        self.num_agents = 2
        self.share_reward = share_reward

        # load dataset
        self.data = data
        self.img_shape = (3, 32, 32)

    def seed(self, seed=1):
        # seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_obs(self):
        return [self.get_agent_obs(i) for i in range(self.num_agents)]

    def get_reward(self, env_act):
        r = [self.get_agent_reward(env_act[i], i
                                   ) for i in range(self.num_agents)]
        if self.share_reward:
            total_r = sum(r)
            r = [total_r for _ in range(self.num_agents)]
        return r

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        img = np.zeros(shape=(100, 100, 3), dtype=np.uint8)
        return img


class MultiStepGame(Game):
    def __init__(self, seed, share_reward, data, discrete_comm, comm_size,
                 debug=False):
        super(MultiStepGame, self).__init__(seed, share_reward, data, debug)

        # (env_act_space, comm_act_space)
        self.discrete_comm = discrete_comm
        self.comm_size = comm_size
        if discrete_comm:
            comm_space = spaces.MultiBinary(comm_size)
        else:
            comm_space = spaces.Box(low=0.0, high=1.0, shape=(comm_size,),
                                    dtype=np.float32)
        self.action_space = spaces.Tuple([spaces.Discrete(11),
                                          comm_space])

        obs_space = {
            'img': spaces.Box(low=0,
                              high=255,
                              shape=self.img_shape,  # (C, H, W)
                              dtype=np.uint8),
            'label': spaces.Discrete(10),
            'comm': comm_space,
            'selfcomm': comm_space,
            't': spaces.Discrete(5),
        }
        self.observation_space = spaces.Dict(obs_space)
        self.max_steps = 4

    def step(self, action):
        rew = [0., 0.]
        done = False
        info = {}
        env_act, comm_act = list(zip(*action))

        # in the first step, both agents send a size-n message
        self.agent_comm = comm_act

        if self.debug:
            print('Time:', self.t)
            print('Agent env act:', env_act)
            print('Agent comm act:', comm_act)

        if self.t >= self.max_steps:
            rew = self.get_reward(env_act)
            done = True

            if self.debug:
                print('Agent r:', rew)

        obs = self.get_obs()
        self.t += 1
        return obs, rew, done, info

    def reset(self):
        self.t = 0
        self.agent_comm = np.zeros((self.num_agents, self.comm_size))
        self.sample_idx = self.np_random.randint(low=0,
                                                 high=len(self.data),
                                                 size=2)
        if self.debug:
            print('Reset Multi-Step MNIST')
            print('Sampled data class labels:',
                  self.data[self.sample_idx[0]][1],
                  self.data[self.sample_idx[1]][1])
        return self.get_obs()

    def get_agent_reward(self, env_act, agent_idx):
        other_idx = int(1 - agent_idx)
        if env_act == self.data[self.sample_idx[other_idx]][1]:
            return 0.5
        else:
            return 0.0

    def get_agent_obs(self, agent_idx):
        return {
            'img': (self.data[self.sample_idx[agent_idx]][0]),
            'label': self.data[self.sample_idx[agent_idx]][1],
            'comm': self.agent_comm[int(1 - agent_idx)],
            'selfcomm': self.agent_comm[agent_idx],
            't': self.t,
        }

    def get_agent_data(self, agent_idx):
        return {
            'img': self.data[self.sample_idx[agent_idx]][0],
            'label': self.data[self.sample_idx[agent_idx]][1],
        }

    def describe(self):
        labels = [self.data[self.sample_idx[0]][1],
                  self.data[self.sample_idx[1]][1]]
        for i in range(self.num_agents):
            print(f'Agent {i} said {self.agent_comm[i]}')
            print(f'Agent {i} sees {classes[labels[i]]} ({labels[i]})')
