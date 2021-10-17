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
from util.video import make_video


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
                 num_eval_episodes=10, net_type=''):
        super().__init__()
        self.master = master
        self.net = net
        self.sleep_duration = sleep_duration
        self.gpu_id = gpu_id
        self.fps = 10
        self.agents = [f'agent_{i}' for i in range(env.num_agents)]

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

        self.net_type = net_type
        return

    @torch.no_grad()
    @within_cuda_device
    def run(self):
        self.master.init_tensorboard()

        while not self.master.is_done():

            weight_iter = self.master.copy_weights(self.net)

            log_dict = defaultdict(int)
            for eval_id in range(self.num_eval_episodes):
                env_copy = copy.deepcopy(self.eval_env[eval_id])

                state = env_copy.reset()
                state_var = ops.to_state_var(state)

                if self.net.is_recurrent:
                    hidden_state = self.net.init_hidden()

                done = False

                frames = []
                env_rewards = []

                ents = []
                agent_steps = [-1 for _ in range(env_copy.num_agents)]

                # mask first (environment) actions after an agent is done
                env_mask_idx = [None for _ in range(len(self.agents))]

                while not check_done(done):
                    if self.net_type == 'ae':
                        plogit, _, hidden_state, comm_out, _ = self.net(
                            state_var, hidden_state, env_mask_idx=env_mask_idx)
                        _, _, ent, action = self.net.take_action(plogit,
                                                                 comm_out)
                    else:
                        plogit, _, hidden_state = self.net(
                            state_var, hidden_state, env_mask_idx=env_mask_idx)
                        action, _, ent = self.net.take_action(plogit)

                    # record action entropy
                    ents.append(ent)

                    state, reward, done, info = env_copy.step(action)
                    state_var = ops.to_state_var(state)

                    for agent_id, a in enumerate(self.agents):
                        if info[a]['done'] and agent_steps[agent_id] < 0:
                            agent_steps[agent_id] = env_copy.step_count

                    for agent_id, a in enumerate(self.agents):
                        if info[a]['done'] and \
                                env_mask_idx[agent_id] is None:
                            env_mask_idx[agent_id] = [0, 1, 2, 3]

                    if hasattr(env_copy, 'get_raw_obs'):
                        frame = env_copy.get_raw_obs()
                        frame = frame[None, ...]
                    else:
                        warnings.warn('environment does not have get_raw_obs() '
                                      + 'assuming state to be an image.')
                        frame = state

                    frames.extend(frame)
                    env_rewards.append(info['rew_by_act'][0])

                # save env action entropy plot
                ent_path = osp.join(self.video_save_dir,
                                    f'latest_ent_{eval_id}.png')
                plot_ents(np.asarray(ents), agent_steps, ent_path,
                          env_copy.adv_indices, max_ent=[np.log(5), np.log(2)])

                # save video
                latest_path = osp.join(self.video_save_dir,
                                       f'latest_{eval_id}.mp4')
                make_video(latest_path, frames, fps=self.fps, verbose=False)

                # only save the first video in history
                if (weight_iter + 1) % self.video_save_freq == 0 \
                        and eval_id == 0:
                    save_path = osp.join(self.video_save_dir,
                                         f'{weight_iter}.png')
                    shutil.copyfile(ent_path, save_path)

                    save_path = osp.join(self.video_save_dir,
                                         f'{weight_iter}.mp4')
                    shutil.copyfile(latest_path, save_path)

                # log info
                max_time = -1
                for i, agent in enumerate(self.agents):
                    agent_rewards = np.array([r[agent] for r in env_rewards])
                    if max_time < 0:
                        max_time = len(agent_rewards)

                    ctime = max_time if agent_steps[i] < 0 \
                        else agent_steps[i]

                    log_dict[f'rewards/{agent}'] += np.sum(agent_rewards)
                    log_dict[f'steps/{agent}'] += ctime

                    # cumulative adv/nonadv info
                    if i in env_copy.adv_indices:
                        log_dict['adv_rew'] += np.sum(agent_rewards)
                        log_dict['adv_steps'] += ctime
                    else:
                        log_dict['nonadv_rew'] += np.sum(agent_rewards)
                        log_dict['nonadv_steps'] += ctime

                log_dict['episode_len'] += max_time
                if info['agent_0']['nonadv_done']:
                    log_dict['timeout'] += 0
                else:
                    log_dict['timeout'] += 1

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
