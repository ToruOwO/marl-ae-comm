from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.a3c_template import A3CTemplate, take_action, take_comm_action
from model.init import normalized_columns_initializer, weights_init
from model.model_utils import LSTMhead, InputProcessingModule


class RichSharedNetwork(A3CTemplate):
    def __init__(self, obs_space, act_space, num_agents, comm_size, comm_len,
                 discrete_comm, hidden_size=256, emb_size=64,
                 num_blind_agents=0, share_critic=False, layer_norm=False,
                 comm_rnn=True):
        super().__init__()

        # assume action space is a Tuple of 2 spaces
        self.env_action_size = act_space[0].n  # Discrete

        # assume comm action space to be Discrete/MultiDiscrete/Box
        if act_space[1].__class__.__name__ == 'MultiDiscrete':
            self.comm_action_size = np.sum(act_space[1].nvec)
        elif act_space[1].__class__.__name__ == 'Discrete':
            self.comm_action_size = act_space[1].n
        elif act_space[1].__class__.__name__ == 'Box':
            self.comm_action_size = act_space[1].shape[0]
        else:
            raise NotImplementedError

        self.comm_action_space = act_space[1]

        self.num_agents = num_agents

        self.input_processor = InputProcessingModule(obs_space,
                                                     comm_size,
                                                     comm_len,
                                                     discrete_comm,
                                                     emb_size,
                                                     num_agents,
                                                     num_blind_agents,
                                                     layer_norm,
                                                     comm_rnn)

        # individual memories
        self.feat_dim = self.input_processor.feat_dim
        self.head = nn.ModuleList(
            [LSTMhead(self.feat_dim - 32 * 3 * 3, hidden_size, num_layers=1
                      ) for _ in range(num_blind_agents)
             ] + [LSTMhead(self.feat_dim, hidden_size, num_layers=1
                           ) for _ in range(num_blind_agents, num_agents)])
        self.is_recurrent = True

        # separate AC for env action and comm action
        self.share_critic = share_critic
        if share_critic:
            self.env_critic_linear = nn.Linear(hidden_size, 1)
            self.comm_critic_linear = nn.Linear(hidden_size, 1)
        else:
            self.env_critic_linear = nn.ModuleList([nn.Linear(
                hidden_size, 1) for _ in range(num_agents)])
            self.comm_critic_linear = nn.ModuleList([nn.Linear(
                hidden_size, 1) for _ in range(num_agents)])
        self.env_actor_linear = nn.ModuleList([nn.Linear(
            hidden_size, self.env_action_size) for _ in range(num_agents)])
        self.comm_actor_linear = nn.ModuleList([nn.Linear(
            hidden_size, self.comm_action_size) for _ in range(num_agents)])

        self.reset_parameters()
        return

    def reset_parameters(self):
        for actor in [self.env_actor_linear, self.comm_actor_linear]:
            for m in actor:
                m.weight.data = normalized_columns_initializer(
                    m.weight.data, 0.01)
                m.bias.data.fill_(0)

        for critic in [self.env_critic_linear, self.comm_critic_linear]:
            if self.share_critic:
                critic.weight.data = normalized_columns_initializer(
                    critic.weight.data, 1.0)
                critic.bias.data.fill_(0)
            else:
                for m in critic:
                    m.weight.data = normalized_columns_initializer(
                        m.weight.data, 1.0)
                    m.bias.data.fill_(0)
        return

    def init_hidden(self):
        return [head.init_hidden() for head in self.head]

    def take_action(self, policy_logits):
        """
        Args:
            policy_logits: a tuple of (env_logits, comm_logits)
        Returns:
            act_dicts: a dict containing all agents' [env_act, comm_act]
        """
        act_dict = {}
        act_logp_dict = {}
        ent_list = []
        env_logits, comm_logits = policy_logits
        for agent_name, envl in env_logits.items():
            comml = comm_logits[agent_name]
            env_act, env_act_logp, env_ent = take_action(
                envl, self.env_action_size)

            comm_act, comm_act_logp, comm_ent = take_comm_action(
                comml, self.comm_action_space)

            act_dict[agent_name] = [env_act, comm_act]
            act_logp_dict[agent_name] = [env_act_logp, comm_act_logp]

            ent_list.append(env_ent)

            # disable rendering of comm ent for now
            # ent_list.append([env_ent, *comm_ent])
        return act_dict, act_logp_dict, ent_list

    def forward(self, inputs, hidden_state=None, env_mask_idx=None):
        assert type(inputs) is dict
        assert len(inputs.keys()) == self.num_agents + 1  # agents + global

        # WARNING: the following code only works for Python 3.6 and beyond

        # (1) pre-process inputs
        cat_feat = self.input_processor(inputs)

        # (2) predict policy and values separately
        env_actor_out, env_critic_out = {}, {}
        comm_actor_out, comm_critic_out = {}, {}

        for i, agent_name in enumerate(inputs.keys()):
            if agent_name == 'global':
                continue

            x, hidden_state[i] = self.head[i](cat_feat[i], hidden_state[i])

            env_actor_out[agent_name] = self.env_actor_linear[i](x)
            comm_actor_out[agent_name] = self.comm_actor_linear[i](x)

            if self.share_critic:
                env_critic_out[agent_name] = self.env_critic_linear(x)
                comm_critic_out[agent_name] = self.comm_critic_linear(x)
            else:
                env_critic_out[agent_name] = self.env_critic_linear[i](x)
                comm_critic_out[agent_name] = self.comm_critic_linear[i](x)

            # mask logits of unavailable actions if provided
            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[agent_name][0, env_mask_idx[i]] = -1e10

        return (env_actor_out, comm_actor_out),\
               (env_critic_out, comm_critic_out), hidden_state
