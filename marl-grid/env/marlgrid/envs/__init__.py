import inspect
import numpy.random as random
import sys
from gym.envs.registration import register as gym_register

from .findgoal import FindGoalMultiGrid
from .redbluedoors import RedBlueDoorsMultiGrid
from ..agents import GridAgentInterface
from ..base import MultiGridEnv

this_module = sys.modules[__name__]


def make_agents(
        n_agents,
        view_size,
        view_tile_size=8,
        view_offset=0,
        agent_color=None,
        observation_style='image',
        observe_position=False,
        observe_self_position=False,
        observe_done=False,
        observe_self_env_act=False,
        observe_t=False,
        restrict_actions=True,
        comm_dim=0,
        comm_len=1,
        discrete_comm=True,
        n_adversaries=0,
        see_through_walls=True,
        neutral_shape=True,
        can_overlap=True,
):
    colors = ['red', 'blue', 'purple', 'orange', 'olive', 'pink']
    assert n_agents <= len(colors)

    if isinstance(view_size, list):
        assert len(view_size) == n_agents
        view_size_lst = view_size
    else:
        view_size_lst = [view_size for _ in range(n_agents)]

    # assign roles differently in each environment
    adv_indices = random.choice([i for i in range(n_agents)],
                                n_adversaries,
                                replace=False)

    agents = [GridAgentInterface(
        color=c if agent_color is None else agent_color,
        neutral_shape=neutral_shape,
        can_overlap=can_overlap,
        view_size=view_size_lst[i],
        view_tile_size=view_tile_size,
        view_offset=view_offset,
        observation_style=observation_style,
        observe_position=observe_position,
        observe_self_position=observe_self_position,
        observe_done=observe_done,
        observe_self_env_act=observe_self_env_act,
        observe_t=observe_t,
        restrict_actions=restrict_actions,
        see_through_walls=see_through_walls,
        comm_dim=comm_dim,
        comm_len=comm_len,
        discrete_comm=discrete_comm,
        n_agents=n_agents,
        is_adversary=1 if i in adv_indices else 0
    ) for i, c in enumerate(colors[:n_agents])]
    return agents


def get_env_creator(
        env_class,
        n_agents,
        grid_size,
        view_size,
        view_tile_size=8,
        view_offset=0,
        agent_color=None,
        observation_style='image',
        observe_position=False,
        observe_self_position=False,
        observe_done=False,
        observe_self_env_act=False,
        observe_t=False,
        restrict_actions=True,
        comm_dim=0,
        comm_len=1,
        discrete_comm=True,
        n_adversaries=0,
        neutral_shape=True,
        can_overlap=True,
        env_kwargs={}
):
    def env_creator(env_config):
        agents = make_agents(n_agents,
                             view_size,
                             view_tile_size=view_tile_size,
                             view_offset=view_offset,
                             agent_color=agent_color,
                             observation_style=observation_style,
                             observe_position=observe_position,
                             observe_self_position=observe_self_position,
                             observe_done=observe_done,
                             observe_self_env_act=observe_self_env_act,
                             observe_t=observe_t,
                             restrict_actions=restrict_actions,
                             comm_dim=comm_dim,
                             comm_len=comm_len,
                             discrete_comm=discrete_comm,
                             n_adversaries=n_adversaries,
                             neutral_shape=neutral_shape,
                             can_overlap=can_overlap)
        env_config.update(env_kwargs)
        env_config['grid_size'] = grid_size
        env_config['agents'] = agents
        return env_class(env_config)

    return env_creator


def get_env_class_creator(
        env_class,
        n_agents,
        grid_size,
        view_size,
        view_tile_size=8,
        view_offset=0,
        agent_color=None,
        observation_style='image',
        observe_position=False,
        observe_self_position=False,
        observe_done=False,
        observe_self_env_act=False,
        observe_t=False,
        restrict_actions=True,
        comm_dim=0,
        comm_len=1,
        discrete_comm=True,
        n_adversaries=0,
        neutral_shape=True,
        can_overlap=True,
        env_kwargs={}
):
    class GymEnv(env_class):
        def __new__(cls):
            agents = make_agents(n_agents,
                                 view_size,
                                 view_tile_size=view_tile_size,
                                 view_offset=view_offset,
                                 agent_color=agent_color,
                                 observation_style=observation_style,
                                 observe_position=observe_position,
                                 observe_self_position=observe_self_position,
                                 observe_done=observe_done,
                                 observe_self_env_act=observe_self_env_act,
                                 observe_t=observe_t,
                                 restrict_actions=restrict_actions,
                                 comm_dim=comm_dim,
                                 comm_len=comm_len,
                                 discrete_comm=discrete_comm,
                                 n_adversaries=n_adversaries,
                                 neutral_shape=neutral_shape,
                                 can_overlap=can_overlap)
            instance = super(env_class, GymEnv).__new__(env_class)
            env_config = env_kwargs
            env_config['grid_size'] = grid_size
            env_config['agents'] = agents

            instance.__init__(config=env_config)
            return instance
    return GymEnv


def register_env(
        env_name,
        n_agents,
        grid_size,
        view_size,
        view_tile_size,
        comm_dim,
        comm_len,
        discrete_comm,
        n_adversaries,
        observation_style,
        observe_position,
        observe_self_position,
        observe_done,
        observe_self_env_act,
        observe_t,
        neutral_shape,
        can_overlap,
        use_gym_env=False,
        env_configs={},
        clutter_density=0.15,
        env_type='c',
):
    if env_type == 'c':
        env_class = FindGoalMultiGrid
        restrict_actions = True
    elif env_type == 'd':
        env_class = RedBlueDoorsMultiGrid
        assert n_agents == 2
        assert n_adversaries == 0
        restrict_actions = False
    else:
        raise ValueError(f'env type {env_type} not supported')

    env_creator = get_env_creator(
        env_class,
        n_agents=n_agents,
        grid_size=grid_size,
        view_size=view_size,
        view_tile_size=view_tile_size,
        comm_dim=comm_dim,
        comm_len=comm_len,
        discrete_comm=discrete_comm,
        n_adversaries=n_adversaries,
        observation_style=observation_style,
        observe_position=observe_position,
        observe_self_position=observe_self_position,
        observe_done=observe_done,
        observe_self_env_act=observe_self_env_act,
        observe_t=observe_t,
        restrict_actions=restrict_actions,
        neutral_shape=neutral_shape,
        env_kwargs={
            'clutter_density': clutter_density,
            'randomize_goal': True,
        }
    )

    # register for Gym
    if use_gym_env:
        gym_env_class = get_env_class_creator(
            env_class,
            n_agents=n_agents,
            grid_size=grid_size,
            view_size=view_size,
            view_tile_size=view_tile_size,
            comm_dim=comm_dim,
            comm_len=comm_len,
            discrete_comm=discrete_comm,
            n_adversaries=n_adversaries,
            observation_style=observation_style,
            observe_position=observe_position,
            observe_self_position=observe_self_position,
            observe_done=observe_done,
            observe_self_env_act=observe_self_env_act,
            observe_t=observe_t,
            restrict_actions=restrict_actions,
            neutral_shape=neutral_shape,
            can_overlap=can_overlap,
            env_kwargs={
                'clutter_density': clutter_density,
                'randomize_goal': True,
            }
        )

        env_class_name = 'env_0'
        setattr(this_module, env_class_name, gym_env_class)
        gym_register(env_name,
                     entry_point=f'marlgrid.envs:{env_class_name}')

    env = env_creator(env_configs)
    return env


def env_from_config(env_config, randomize_seed=True):
    possible_envs = {k: v for k, v in globals().items() if
                     inspect.isclass(v) and issubclass(v, MultiGridEnv)}

    env_class = possible_envs[env_config['env_class']]

    env_kwargs = {k: v for k, v in env_config.items() if k != 'env_class'}
    if randomize_seed:
        env_kwargs['seed'] = env_kwargs.get('seed', 0
                                            ) + random.randint(0, 1337 * 1337)

    return env_class(**env_kwargs)
