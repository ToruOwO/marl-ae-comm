from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from easydict import EasyDict as edict
import marlgrid.envs
import marlgrid


def create_grid_world_env(env_cfg):
    """
    Automatically generate env instance from env configs.
    """
    env_name = get_env_name(env_cfg)

    env = marlgrid.envs.register_env(
        env_name=env_name,
        n_agents=env_cfg.num_agents,
        grid_size=env_cfg.grid_size,
        view_size=env_cfg.view_size,
        view_tile_size=env_cfg.view_tile_size,
        comm_dim=2,
        comm_len=env_cfg.comm_len,
        discrete_comm=env_cfg.discrete_comm,
        n_adversaries=0,
        observation_style=env_cfg.observation_style,
        observe_position=env_cfg.observe_position,
        observe_self_position=env_cfg.observe_self_position,
        observe_done=env_cfg.observe_done,
        observe_self_env_act=env_cfg.observe_self_env_act,
        observe_t=env_cfg.observe_t,
        neutral_shape=env_cfg.neutral_shape,
        can_overlap=env_cfg.can_overlap,
        use_gym_env=False,
        env_configs={
            'max_steps': env_cfg.max_steps,
            'team_reward_multiplier': env_cfg.team_reward_multiplier,
            'team_reward_type': env_cfg.team_reward_type,
            'team_reward_freq': env_cfg.team_reward_freq,
            'seed': env_cfg.seed,
            'active_after_done': env_cfg.active_after_done,
            'discrete_position': env_cfg.discrete_position,
            'separate_rew_more': env_cfg.separate_rew_more,
            'info_gain_rew': env_cfg.info_gain_rew,
        },
        clutter_density=env_cfg.clutter_density)

    return env


def get_env_name(env_cfg):
    """
    Automatically generate env name from env configs.
    """
    assert env_cfg.env_type == 'c'
    name = f'MarlGrid-{env_cfg.num_agents}Agent'

    if env_cfg.comm_len > 0:
        name += f'{env_cfg.comm_len}C'
        if not env_cfg.discrete_comm:
            name += 'cont'

    if env_cfg.view_size != 7:
        name += f'{env_cfg.view_size}Vs'

    name += f'{env_cfg.grid_size}x{env_cfg.grid_size}-v0'
    return name
