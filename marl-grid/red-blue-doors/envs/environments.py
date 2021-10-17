from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import gym
from .grid_world_environment import create_grid_world_env
from .wrappers import DictObservationNormalizationWrapper, \
    GridWorldEvaluatorWrapper


def make_environment(env_cfg, lock=None):
    """ Use this to make Environments """

    env_name = env_cfg.env_name

    assert env_name.startswith('MarlGrid')
    env = create_grid_world_env(env_cfg)
    env = GridWorldEvaluatorWrapper(env)
    env = DictObservationNormalizationWrapper(env)

    return env
