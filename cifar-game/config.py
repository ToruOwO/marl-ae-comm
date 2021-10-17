from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
import datetime
import argparse

import json
from easydict import EasyDict as edict

from util import CfgNode
from util.misc import cprint


def get_env_cfg():
    config = CfgNode()

    config.seed = 1
    config.share_reward = True
    config.discrete_comm = False
    config.comm_size = 1

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
    config.train_iter = 300000

    config.lr = 0.0001

    # experiment id
    config.resume_path = ''

    # the policy head
    config.policy = 'lstm'  # ['fc', 'lstm']
    config.model = 'shared'  # ['shared']
    config.share_critic = False
    config.layer_norm = True

    # comm options
    config.comm_type = 1  # 0 - no commm, 1 - non-rl comm, 2 - rl comm
    config.aux_loss = ''  # ['a', 'ap', '']
    config.ae_fc_size = 0
    config.use_mlp = False
    config.hidden_size = 128
    config.img_feat_size = 10
    config.comm_pg = True

    # mask logits of unavailable actions
    config.mask = True

    # eval configs
    config.video_save_freq = 20
    config.ckpt_save_freq = 100
    config.num_eval_episodes = 10
    config.num_eval_videos = 10

    config.debug = False

    # ===
    # set configs with config file and list options
    # ===

    set_params(config, args.config_path, args.set)

    # ===
    # automatically generate exp name based on configs
    # ===

    curr_time = str(datetime.datetime.now())[:16].replace(' ', '_')

    id_args = [['seed', config.env_cfg.seed],
               ['lr', config.lr],
               ['tmax', config.tmax],
               ['workers', config.num_workers],
               ['comm', config.comm_type],
               ['clen', config.env_cfg.comm_size],
               ['cpg', config.comm_pg]]

    if config.comm_type == 1:
        id_args += [['aux', config.aux_loss], ['aefc', config.ae_fc_size]]

    if config.comm_type == 1 or config.comm_type == 3:
        id_args += [['mlp', config.use_mlp]]

    cfg_id = '_'.join([f'{n}-{v}' for n, v in id_args])

    if config.env_cfg.discrete_comm:
        cfg_id += '_disc'
    else:
        cfg_id += '_cont'

    if config.id:
        cfg_id = '{}_{}'.format(config.id, cfg_id)

    if eval:
        cfg_id += '_eval'

    exp_name = 'a3c_{}_{}'.format(cfg_id, curr_time)

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
