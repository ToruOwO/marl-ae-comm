from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, os.path as osp
import sys
import time

import config
from envs.environments import make_environment
from actor_critic.master import Master
from actor_critic.worker import Worker
from actor_critic.evaluator import Evaluator
from model.rich import RichSharedNetwork
from model.hard import HardSharedNetwork
from util.shared_opt import SharedAdam


if __name__ == '__main__':
    # (0) args and steps to make this work.
    # Disable the python spawned processes from using multiple threads.
    print(torch.multiprocessing.get_start_method())
    mp.set_start_method('spawn', force=True)
    os.environ['OMP_NUM_THREADS'] = '1'

    cfg = config.parse()

    save_dir_fmt = osp.join(f'./{cfg.run_dir}', cfg.exp_name + '/{}')
    print('>> {}'.format(cfg.exp_name))

    # (1) create environment
    create_env = lambda: make_environment(cfg.env_cfg)
    env = create_env()

    if cfg.env_cfg.observation_style == 'dict' and cfg.env_cfg.comm_len <= 0:
        create_net = lambda: HardSharedNetwork(
            obs_space=env.observation_space,
            action_size=env.action_space.n,
            num_agents=cfg.env_cfg.num_agents,
            num_blind_agents=cfg.env_cfg.num_blind_agents,
            share_critic=cfg.share_critic,
            layer_norm=cfg.layer_norm)
    elif cfg.env_cfg.observation_style == 'dict':
        create_net = lambda: RichSharedNetwork(
            obs_space=env.observation_space,
            act_space=env.action_space,
            num_agents=cfg.env_cfg.num_agents,
            comm_size=2,
            comm_len=cfg.env_cfg.comm_len,
            discrete_comm=cfg.env_cfg.discrete_comm,
            num_blind_agents=cfg.env_cfg.num_blind_agents,
            share_critic=cfg.share_critic,
            layer_norm=cfg.layer_norm,
            comm_rnn=cfg.comm_rnn)
    else:
        raise ValueError('Observation style {} not supported'.format(
            cfg.env_cfg.observation_style))

    # (2) create master network.
    # hogwild-style update will be applied to the master weight.
    master_lock = mp.Lock()
    net = create_net()
    net.share_memory()

    opt = SharedAdam(net.parameters(), lr=cfg.lr)

    if cfg.resume_path:
        ckpt = torch.load(cfg.resume_path)
        global_iter = mp.Value('i', ckpt['iter'])
        net.load_state_dict(ckpt['net'])
        opt.load_state_dict(ckpt['opt'])
        print('>>>>> Loaded ckpt from iter', ckpt['iter'])
    else:
        global_iter = mp.Value('i', 0)
    global_done = mp.Value('i', 0)

    master = Master(net, opt, global_iter, global_done, master_lock,
                    writer_dir=save_dir_fmt.format('tb'),
                    max_iteration=cfg.train_iter)

    # (3) create workers
    num_acts = 1
    if cfg.env_cfg.comm_len > 0:
        num_acts = 2

    workers = []
    for worker_id in range(cfg.num_workers):
        gpu_id = cfg.gpu[worker_id % len(cfg.gpu)]
        print(f'(worker {worker_id}) initializing on gpu {gpu_id}')

        with torch.cuda.device(gpu_id):
            workers += [Worker(master,
                               create_net().cuda(),
                               create_env(),
                               worker_id=worker_id,
                               gpu_id=gpu_id,
                               num_acts=num_acts,
                               anneal_comm_rew=cfg.anneal_comm_rew),]

    # (4) create a separate process to dump latest result (optional)
    eval_gpu_id = cfg.gpu[-1]

    with torch.cuda.device(eval_gpu_id):
        evaluator = Evaluator(master, create_net().cuda(), create_env(),
                              save_dir_fmt=save_dir_fmt,
                              gpu_id=eval_gpu_id,
                              sleep_duration=10,
                              video_save_freq=cfg.video_save_freq,
                              ckpt_save_freq=cfg.ckpt_save_freq,
                              num_eval_episodes=cfg.num_eval_episodes)
        workers.append(evaluator)

    # (5) start training!

    # > start the processes
    [w.start() for w in workers]

    # > join when done
    [w.join() for w in workers]

    master.save_ckpt(cfg.train_iter,
                     osp.join(save_dir_fmt.format('ckpt'), 'latest.pth'))
