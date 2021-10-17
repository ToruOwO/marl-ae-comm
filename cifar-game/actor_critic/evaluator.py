import csv
import copy
import time
import os
import os.path as osp
import shutil
import warnings
from collections import defaultdict

import numpy as np

import torch
import torch.multiprocessing as mp

from util import ops
from util.decorator import within_cuda_device
from util.eval_util import plot_ents
from util.misc import check_done


class Evaluator(mp.Process):
    """
    Evaluation process. The process will generate new video every
    sleep_duration.

    Args:
        master: master network instance.
        net: network with same architecture as the master network
        env: environment
        save_dir_fmt: path format to dump videos.
        gpu_id: the cuda gpu device id used to initialize variables. the
            `within_cuda_device` decorator uses this.
        sleep_duration: amount of seconds to wait until the next log.
    """

    def __init__(self, master, net, env, save_dir_fmt, gpu_id=0,
                 sleep_duration=10, video_save_freq=10, ckpt_save_freq=10,
                 num_eval_episodes=10):
        super().__init__()
        self.master = master
        self.net = net
        self.sleep_duration = sleep_duration
        self.gpu_id = gpu_id
        self.fps = 10
        self.num_agents = env.num_agents

        self.num_eval_episodes = num_eval_episodes
        self.video_save_dir = save_dir_fmt.format('video')
        self.video_save_freq = video_save_freq
        self.ckpt_save_dir = save_dir_fmt.format('ckpt')
        self.ckpt_save_freq = ckpt_save_freq

        # make fixed copies of environment for consistent evaluation
        self.eval_env = []
        for _ in range(num_eval_episodes):
            env_copy = copy.deepcopy(env)
            self.eval_env.append(env_copy)
            env.reset()

        os.makedirs(self.video_save_dir, exist_ok=True)
        os.makedirs(self.ckpt_save_dir, exist_ok=True)

        # log first row info
        csv_path = osp.join(self.video_save_dir, 'train_log.csv')
        row = ['weight_iter', 'eval_id', 'reward', 'action', 'label_0',
               'label_1', 'comm_0', 'comm_1']
        with open(csv_path, 'w') as f:
            w = csv.writer(f)
            w.writerow(row)

    @torch.no_grad()
    @within_cuda_device
    def run(self):
        self.master.init_tensorboard()

        while not self.master.is_done():

            weight_iter = self.master.copy_weights(self.net)

            log_dict = {}
            rewards = []
            for eval_id in range(self.num_eval_episodes):
                env_copy = copy.deepcopy(self.eval_env[eval_id])

                state = env_copy.reset()
                state_var = ops.to_state_var(state)

                if self.net.is_recurrent:
                    hidden_state = self.net.init_hidden()

                done = False

                ents = []

                env_mask_idx = [None for _ in range(self.num_agents)]

                # log comm
                comm = []
                while not check_done(done):
                    plogit, _, hidden_state, comm_out, _ = self.net(
                        state_var, hidden_state, env_mask_idx=env_mask_idx)
                    action, _, ent, all_actions = self.net.take_action(plogit,
                                                                       comm_out)

                    # record action entropy
                    ents.append(ent)

                    state, reward, done, info = env_copy.step(all_actions)
                    state_var = ops.to_state_var(state)

                    comm.append(env_copy.agent_comm)

                # save env action entropy plot
                labels = (env_copy.data[env_copy.sample_idx[0]][1],
                          env_copy.data[env_copy.sample_idx[1]][1])
                comm = list(zip(*comm))
                ent_path = osp.join(self.video_save_dir,
                                    f'latest_ent_{eval_id}.png')
                plot_ents(np.asarray(ents), labels, action, comm, ent_path,
                          max_ent=[np.log(11), np.log(2)])

                # only record the last-step reward (all other steps are zero)
                rewards.append(reward)

                # log game info
                csv_path = osp.join(self.video_save_dir, 'train_log.csv')
                if env_copy.share_reward:
                    r = reward[0]
                else:
                    r = reward
                row = [weight_iter, eval_id, r, action, labels[0], labels[1],
                       comm[0], comm[1]]
                with open(csv_path, 'a') as f:
                    w = csv.writer(f)
                    w.writerow(row)

            # log info
            agent_rewards = list(zip(*rewards))

            if env_copy.share_reward:
                # only log 1 agent
                log_dict['rewards'] = np.sum(agent_rewards[0])
            else:
                for i in range(self.num_agents):
                    log_dict[f'rewards/{i}'] = np.sum(agent_rewards[i])

            # average logged info
            for k, v in log_dict.items():
                self.master.writer.add_scalar(k, v / self.num_eval_episodes,
                                              weight_iter)

            # save weights
            self.master.save_ckpt(weight_iter,
                                  osp.join(self.ckpt_save_dir, 'latest.pth'))

            if (weight_iter + 1) % self.ckpt_save_freq == 0:
                self.master.save_ckpt(weight_iter,
                                      osp.join(self.ckpt_save_dir,
                                               f'{weight_iter}.pth'))

            time.sleep(self.sleep_duration)

        print('evaluator is done.')
        return
