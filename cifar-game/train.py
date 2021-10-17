"""Training script for CIFAR Game environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets
import torchvision.transforms as transforms

import numpy as np
import os
import os.path as osp
import sys
import time

import config
from envs.game_environment import create_game_env
from actor_critic.master import Master
from actor_critic.worker import Worker
from actor_critic.worker_pg import WorkerPGComm
from actor_critic.evaluator import Evaluator
from model.cifar import CifarNet
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

    # (1) load data and create environment
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = datasets.CIFAR10(root='./data',
                            train=True,
                            download=True,
                            transform=transform)

    create_env = lambda: create_game_env(cfg.env_cfg, data)
    env = create_env()

    create_net = lambda: CifarNet(
        act_space=env.action_space,
        comm_type=cfg.comm_type,
        comm_pg=cfg.comm_pg,
        aux_loss=cfg.aux_loss,
        img_feat_size=cfg.img_feat_size,
        hidden_size=cfg.hidden_size,
        comm_size=cfg.env_cfg.comm_size,
        ae_fc_size=cfg.ae_fc_size,
        use_mlp=cfg.use_mlp,
        debug=cfg.debug)

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

    # (3) create slave workers
    workers = []
    for worker_id in range(cfg.num_workers):
        gpu_id = cfg.gpu[worker_id % len(cfg.gpu)]
        print(f'(worker {worker_id}) initializing on gpu {gpu_id}')

        with torch.cuda.device(gpu_id):
            if cfg.comm_type == 2:
                workers += [WorkerPGComm(master,
                                         create_net().cuda(),
                                         create_env(),
                                         worker_id=worker_id,
                                         gpu_id=gpu_id), ]
            else:
                workers += [Worker(master,
                                   create_net().cuda(),
                                   create_env(),
                                   worker_id=worker_id,
                                   gpu_id=gpu_id), ]

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

    # (5) start training

    # > start the processes
    [w.start() for w in workers]

    # > join when done
    [w.join() for w in workers]

    master.save_ckpt(cfg.train_iter,
                     osp.join(save_dir_fmt.format('ckpt'), 'latest.pth'))
