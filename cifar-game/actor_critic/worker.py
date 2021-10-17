from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time, os, sys
from collections import deque

import numpy as np

import torch
import torch.multiprocessing as mp

from loss import policy_gradient_loss

from util import ops
from util.misc import check_done
from util.decorator import within_cuda_device


class Worker(mp.Process):
    """
    A3C worker. Each worker is responsible for collecting data from the
    environment and updating the master network by supplying the gradients.
    The worker re-synchronizes the weight at ever iteration.

    Args:
        master: master network instance.
        net: network with same architecture as the master network
        env: environment
        worker_id: worker id. used for tracking and debugging.
        gpu_id: the cuda gpu device id used to initialize variables. the
            `within_cuda_device` decorator uses this.
        t_max: maximum number of steps to take before applying gradient update.
            Default: `20`
        use_gae: uses generalized advantage estimation.
            Default: `True`
        gamma: hyperparameter for the reward decay.
            Default: `0.99`
        tau: gae hyperparameter.
    """

    def __init__(self, master, net, env, worker_id, gpu_id=0, t_max=20,
                 use_gae=True, gamma=0.99, tau=1.0):
        super().__init__()

        self.worker_id = worker_id
        self.net = net
        self.env = env
        self.master = master
        self.t_max = t_max
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau
        self.gpu_id = gpu_id
        self.reward_log = deque(maxlen=10)  # track last 10 finished rewards
        self.pfmt = 'policy loss: {:0.4f} value loss: {:0.4f} ' + \
                    'entropy loss: {:0.4f} comm loss: {:0.4f} reward: {:0.4f}'
        self.num_agents = env.num_agents

    @within_cuda_device
    def get_trajectory(self, hidden_state, state_var, done):
        """
        extracts a trajectory using the current policy.

        Args:
            hidden_state: last hidden state observed
            state_var: last state observed
            done: boolean value to determine whether the env should be reset
        Returns:
            trajectory: (pi, a, v, r) trajectory [state is not tracked]
            values: reversed trajectory values used for GAE.
            target_value: the last time-step value
            done: updated indicator
        """
        # mask environment actions before the last step
        env_mask_idx = [[i for i in range(10)] for _ in range(self.num_agents)]

        trajectory = []

        while not check_done(done) and len(trajectory) < self.t_max:
            # mask unavailable env actions after individual done
            if self.env.t >= self.env.max_steps:
                env_mask_idx = [[10] for _ in range(self.num_agents)]

            plogit, value, hidden_state, comm_out, comm_loss = self.net(
                state_var, hidden_state, env_mask_idx=env_mask_idx)
            action, _, _, all_actions = self.net.take_action(plogit, comm_out)
            state, reward, done, info = self.env.step(all_actions)
            state_var = ops.to_state_var(state)

            trajectory.append((plogit, action, value, reward, comm_loss))

        # end condition
        if check_done(done):
            target_value = [0 for _ in range(self.num_agents)]
        else:
            with torch.no_grad():
                target_value = self.net(state_var,
                                        hidden_state,
                                        env_mask_idx=env_mask_idx)[1]

        #  Compute Loss: accumulate rewards and compute gradient
        values = [None for _ in range(self.num_agents)]
        if self.use_gae:
            for k in range(self.num_agents):
                values[k] = [x[k] for x in list(zip(*trajectory))[2]]
                values[k].append(ops.to_torch([target_value[k]]))
                values[k].reverse()

        return trajectory, values, target_value, done

    @within_cuda_device
    def run(self):
        self.master.init_tensorboard()
        done = True
        reward_log = 0.

        while not self.master.is_done():
            # synchronize network parameters
            weight_iter = self.master.copy_weights(self.net)
            self.net.zero_grad()

            # reset environment if new episode
            if check_done(done):
                state = self.env.reset()
                state_var = ops.to_state_var(state)
                hidden_state = None

                if self.net.is_recurrent:
                    hidden_state = self.net.init_hidden()

                done = False

                self.reward_log.append(reward_log)
                reward_log = 0.

            # extract trajectory
            trajectory, values, target_value, done = \
                self.get_trajectory(hidden_state, state_var, done)

            all_pls, all_vls, all_els = [], [], []
            all_cls = []

            # compute loss for each action
            loss = 0

            traj = trajectory
            val = values
            tar_val = target_value

            # compute loss - computed backward
            traj.reverse()

            for agent_id in range(self.num_agents):
                gae = torch.zeros(1, 1).cuda()
                t_value = tar_val[agent_id]

                pls, vls, els = [], [], []
                cls = []
                for i, (pi_logit, action, value, reward, comm_loss
                        ) in enumerate(traj):

                    # clip reward (optional)
                    if False:
                        reward = float(np.clip(reward, -1.0, 1.0))

                    # Agent A3C Loss
                    t_value = reward[agent_id] + self.gamma * t_value
                    advantage = t_value - value[agent_id]

                    if self.use_gae:
                        # Generalized advantage estimation (GAE)
                        delta_t = reward[agent_id] + \
                                  self.gamma * val[agent_id][i].data - \
                                  val[agent_id][i + 1].data
                        gae = gae * self.gamma * self.tau + delta_t
                    else:
                        gae = False

                    tl, (pl, vl, el) = policy_gradient_loss(
                        pi_logit[agent_id], action[agent_id],
                        advantage, gae=gae)
                    tl += comm_loss

                    pls.append(ops.to_numpy(pl))
                    vls.append(ops.to_numpy(vl))
                    els.append(ops.to_numpy(el))
                    cls.append(ops.to_numpy(comm_loss))

                    loss += tl

                    if agent_id == 0:
                        # only track one agent's reward
                        reward_log += reward[agent_id]

                all_pls.append(np.mean(pls))
                all_vls.append(np.mean(vls))
                all_els.append(np.mean(els))
                all_cls.append(np.mean(cls))

            # accmulate gradient locally
            loss.backward()

            # log training info to tensorboard
            if self.worker_id == 0:
                log_dict = {}
                for agent_id in range(self.num_agents):
                    log_dict[f'policy_loss/{agent_id}'] = all_pls[agent_id]
                    log_dict[f'value_loss/{agent_id}'] = all_vls[agent_id]
                    log_dict[f'entropy/{agent_id}'] = all_els[agent_id]
                    log_dict[f'comm_loss/{agent_id}'] = all_cls[agent_id]
                log_dict['policy_loss'] = np.mean(all_pls)
                log_dict['value_loss'] = np.mean(all_vls)
                log_dict['entropy'] = np.mean(all_els)
                log_dict['comm_loss'] = np.mean(all_cls)

                for k, v in log_dict.items():
                    self.master.writer.add_scalar(k, v, weight_iter)

            # all_pls, all_vls, all_els shape == (num_agents)
            progress_str = self.pfmt.format(
                np.mean(all_pls),
                np.mean(all_vls),
                np.mean(all_els),
                np.mean(all_cls),
                np.mean(self.reward_log))

            self.master.apply_gradients(self.net)
            self.master.increment(progress_str)

        print(f'worker {self.worker_id} is done.')
        return
