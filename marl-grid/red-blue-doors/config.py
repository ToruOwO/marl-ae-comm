from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
import datetime
import argparse

import json
from easydict import EasyDict as edict

from envs.grid_world_environment import get_env_name
from util import CfgNode
from util.misc import cprint


def get_env_cfg():
    config = CfgNode()

    config.seed = 1

    config.env_type = 'd'

    config.num_agents = 2
    config.num_adversaries = 0

    config.max_steps = 2048
    config.grid_size = 10
    config.observation_style = 'dict'
    config.observe_position = False
    config.observe_self_position = False
    config.observe_self_env_act = False
    config.observe_t = False
    config.observe_done = False
    config.neutral_shape = True
    config.can_overlap = False
    config.active_after_done = False

    # allow agents to observe door state and pos
    config.observe_door = False

    config.discrete_position = True

    config.view_size = 5
    config.view_tile_size = 8
    config.clutter_density = 0.15

    # if `num_blind_agents` == b, the FIRST b agents do not get image obs
    config.num_blind_agents = 0

    # agent comm length
    config.comm_len = 0

    # if False, use continuous communication
    config.discrete_comm = False

    config.team_reward_type = 'none'
    config.team_reward_freq = 'none'
    config.team_reward_multiplier = 1

    return config


def get_config(args, eval=False):
    # ===
    # get default configs
    # ===

    config = CfgNode()

    config.env_cfg = get_env_cfg()

    config.run_dir = 'runs'
    config.num_workers = 16
    config.gpu = [int(g) for g in args.gpu]

    # the prefix to the log
    config.id = ''

    # async update steps
    config.tmax = 20

    # max total training iterations
    config.train_iter = 500000

    config.lr = 0.0001

    # experiment id
    config.resume_path = ''

    # the policy head
    config.policy = 'lstm'  # ['fc', 'lstm']
    config.model = 'shared'  # ['shared']
    config.share_critic = False
    config.layer_norm = True
    config.comm_rnn = True

    # mask logits of unavailable actions
    config.mask = True

    # training
    config.anneal_comm_rew = False
    config.ae_loss_k = 1.0
    config.ae_pg = 0
    config.ae_type = ''  # ['', 'fc', 'mlp', 'rfc', 'rmlp']
    config.img_feat_dim = 64
    config.comm_vf = False

    # eval configs
    config.video_save_freq = 20
    config.ckpt_save_freq = 100
    config.num_eval_episodes = 10
    config.num_eval_videos = 10
    config.eval_ae = False

    # ===
    # set configs with config file and list options
    # ===

    set_params(config, args.config_path, args.set)

    assert not (config.env_cfg.num_adversaries > 0 and
                config.env_cfg.num_blind_agents > 0)

    if config.env_cfg.observe_position or config.env_cfg.observe_done or \
        config.env_cfg.observe_self_position:
        if config.env_cfg.observation_style != 'dict':
            cprint('AUTO: correcting observation_style to _dict_', 'r')
            config.env_cfg.observation_style = 'dict'

    assert config.env_cfg.num_blind_agents <= config.env_cfg.num_agents
    assert config.env_cfg.num_adversaries == 0
    if config.env_cfg.active_after_done and config.mask:
        raise ValueError('active_after_done and mask cannot both be True')

    if (config.env_cfg.observe_position and
            config.env_cfg.observe_self_position):
        raise ValueError('observe_position and observe_self_position cannot '
                         'both be True')

    # ===
    # automatically generate env name based on env configs
    # ===
    config.env_cfg.env_name = get_env_name(config.env_cfg)

    # ===
    # automatically generate exp name based on configs
    # ===

    curr_time = str(datetime.datetime.now())[:16].replace(' ', '_')

    id_args = [['seed', config.env_cfg.seed],
               ['lr', config.lr],
               ['tmax', config.tmax],
               ['workers', config.num_workers],
               ['ms', config.env_cfg.max_steps],
               ['ae_type', config.ae_type]]

    if config.comm_vf:
        id_args += [['commvf', 'True']]

    if config.ae_pg:
        id_args += [['ae_pg', config.ae_pg]]

    if config.img_feat_dim != 64:
        id_args += [['imgdim', config.img_feat_dim]]

    cfg_id = '_'.join([f'{n}-{v}' for n, v in id_args])

    if config.id:
        cfg_id = '{}_{}'.format(config.id, cfg_id)

    if eval:
        cfg_id += '_eval'

    exp_name = '{}/a3c_{}_{}'.format(config.env_cfg.env_name, cfg_id, curr_time)

    config.exp_name = exp_name

    return config


def parse(eval=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--set', nargs='+')
    parser.add_argument('--gpu', nargs='+', default='0',
                        help='specify GPUs. (e.g. `0` or `0 1`)')

    args = parser.parse_args()

    cfg = get_config(args, eval=eval)
    freeze(cfg, save_file=True)

    # print config
    cprint(f'Registered env [{cfg.env_cfg.env_name}]', 'g')
    cprint('------ A3C configurations ------', 'y')
    for arg in sorted(vars(args)):
        print('{0}: {1}'.format(arg, getattr(args, arg)))
    cprint('--------------------------------', 'y')

    return cfg


def set_params(config, file_path=None, list_opt=None):
    """
    Set config parameters with config file and options.
    Option list (usually from command line) has the highest
    overwrite priority.
    """
    if file_path:
        # if list_opt is None or 'run_dir' not in list_opt[::2]:
        #     raise ValueError('Must specify new run directory.')
        print('- Import config from file {}.'.format(file_path))
        config.merge_from_file(file_path)
    if list_opt:
        print('- Overwrite config params {}.'.format(str(list_opt[::2])))
        config.merge_from_list(list_opt)
    return config


def freeze(config, save_file=False):
    """Freeze configuration and save to file (optional)."""
    config.freeze()
    if save_file:
        if not os.path.isdir(config.run_dir):
            os.makedirs(config.run_dir)

        save_dir = os.path.join(config.run_dir, config.exp_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as fout:
            fout.write(config.dump())
