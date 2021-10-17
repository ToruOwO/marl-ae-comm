import copy
import json
import time
import os
import os.path as osp
import shutil
import warnings
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch

from util import ops
from util.decorator import within_cuda_device
from util.misc import check_done


def plot_ents(ents, labels, env_acts, comm, save_path, max_ent=None):
    plt.clf()

    t = ents.shape[0]
    n = ents.shape[1]
    x = np.arange(1, t + 1)

    title = f'labels={labels}, env_acts={env_acts}\n{comm}'

    if len(ents.shape) == 2:
        # num_act == 1
        fig = plt.gcf()
        fig.set_size_inches(10, 5)

        for aid in range(n):
            plt.plot(x, ents[:, aid].flatten(), label=f'Agent{aid}')

        ax = plt.gca()
        if max_ent is not None:
            ax.set_ylim([-0.2, max_ent[0] + 0.2])
            ax.hlines(y=max_ent[0], xmin=0, xmax=t, colors='r', linestyles='--')
        plt.xlabel('t')
        plt.ylabel('ent')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title(title)

    else:
        plt.cla()
        # (T, N, 1 + comm_len)
        assert len(ents.shape) == 3

        fig, axs = plt.subplots(ents.shape[-1],
                                figsize=(10, ents.shape[-1] * 5))

        for i in range(ents.shape[-1]):

            for aid in range(n):
                axs[i].plot(x, ents[:, aid, i].flatten(), label=f'Agent{aid}')

            if i == 0:
                axs[i].set_title('env-act')
            else:
                axs[i].set_title(f'comm-act-{i}')

            axs[i].set(xlabel='t', ylabel='ent')
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            if max_ent is not None:
                if i >= len(max_ent):
                    max_ent_idx = len(max_ent) - 1
                else:
                    max_ent_idx = i
                axs[i].set_ylim([-0.2, max_ent[max_ent_idx] + 0.2])
                axs[i].hlines(y=max_ent[max_ent_idx], xmin=0, xmax=t,
                              colors='r', linestyles='--')

        fig.suptitle(title)

    fig.tight_layout()
    plt.savefig(save_path)


class EvalInterface:
    """
    Evaluation interface.
    """
    def __init__(self, net, env, save_dir_fmt, num_eval_videos=10,
                 num_eval_episodes=10):
        super().__init__()

        net.eval()
        self.net = net
        self.fps = 10
        self.agents = [f'agent_{i}' for i in range(env.num_agents)]

        assert num_eval_videos <= num_eval_episodes

        self.num_eval_videos = num_eval_videos
        self.num_eval_episodes = num_eval_episodes
        self.video_save_dir = save_dir_fmt.format('video')

        # make fixed copies of environment for consistent evaluation
        self.eval_env = []
        for _ in range(num_eval_episodes):
            env_copy = copy.deepcopy(env)
            self.eval_env.append(env_copy)
            env.reset()

        os.makedirs(self.video_save_dir, exist_ok=True)

    def run(self):
        log_dict = {}
        rewards = []
        for eval_id in range(self.num_eval_episodes):
            env_copy = copy.deepcopy(self.eval_env[eval_id])

            state = env_copy.reset()
            state_var = ops.to_state_var(state)

            hidden_state = None

            if self.net.is_recurrent:
                with torch.no_grad():
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
            log_dict[k] = v / self.num_eval_episodes

        json_path = osp.join(self.video_save_dir, f'eval_{eval_id + 1}.json')
        with open(json_path, 'w') as f:
            json.dump(log_dict, f)

        print(log_dict)
