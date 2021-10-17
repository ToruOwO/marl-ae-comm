from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.a3c_template import A3CTemplate, take_action
from model.init import normalized_columns_initializer, weights_init
from model.model_utils import LSTMhead, InputProcessingModule


class HardSharedNetwork(A3CTemplate):
    """
    A network to handle input with no communication.
    """
    def __init__(self, obs_space, action_size, num_agents, hidden_size=256,
                 num_blind_agents=0, share_critic=False, layer_norm=False,
                 observe_door=False):
        super().__init__()

        self.action_size = action_size
        self.num_agents = num_agents
        self.input_processor = InputProcessingModule(
            obs_space,
            comm_size=0,
            comm_len=0,
            discrete_comm=True,
            emb_size=0,
            num_agents=num_agents,
            num_blind_agents=num_blind_agents,
            layer_norm=layer_norm,
            observe_door=observe_door)
        self.feat_dim = self.input_processor.feat_dim

        # individual memories
        self.head = nn.ModuleList(
            [LSTMhead(self.feat_dim - 32 * 3 * 3, hidden_size, num_layers=1
                      ) for _ in range(num_blind_agents)
             ] + [LSTMhead(self.feat_dim, hidden_size, num_layers=1
                           ) for _ in range(num_blind_agents, num_agents)])
        self.is_recurrent = True

        self.share_critic = share_critic
        if share_critic:
            self.critic_linear = nn.Linear(hidden_size, 1)
        else:
            self.critic_linear = nn.ModuleList([nn.Linear(
                hidden_size, 1) for _ in range(num_agents)])
        self.actor_linear = nn.ModuleList([nn.Linear(
            hidden_size, self.action_size) for _ in range(num_agents)])

        self.reset_parameters()
        return

    def reset_parameters(self):
        if self.share_critic:
            self.critic_linear.weight.data = normalized_columns_initializer(
                self.critic_linear.weight.data, 1.0)
            self.critic_linear.bias.data.fill_(0)
        else:
            for m in self.critic_linear:
                m.weight.data = normalized_columns_initializer(m.weight.data,
                                                               1.0)
                m.bias.data.fill_(0)

        for m in self.actor_linear:
            m.weight.data = normalized_columns_initializer(m.weight.data,
                                                           0.01)
            m.bias.data.fill_(0)
        return

    def init_hidden(self):
        return [head.init_hidden() for head in self.head]

    def take_action(self, policy_logit):
        act_dict = {}
        act_logp_dict = {}
        ent_list = []
        for agent_name, logits in policy_logit.items():
            act, act_logp, ent = super(HardSharedNetwork, self).take_action(
                logits)

            act_dict[agent_name] = act
            act_logp_dict[agent_name] = act_logp
            ent_list.append(ent)
        return act_dict, act_logp_dict, ent_list

    def forward(self, inputs, hidden_state=None, env_mask_idx=None):
        assert type(inputs) is dict
        assert len(inputs.keys()) == self.num_agents + 1  # agents + global

        # WARNING: the following code only works for Python 3.6 and beyond

        # (1) pre-process inputs
        cat_feat = self.input_processor(inputs)

        # (2) predict policy and values separately
        env_actor_out, env_critic_out = {}, {}

        for i, agent_name in enumerate(inputs.keys()):
            if agent_name == 'global':
                continue

            x, hidden_state[i] = self.head[i](cat_feat[i], hidden_state[i])

            env_actor_out[agent_name] = self.actor_linear[i](x)

            if self.share_critic:
                env_critic_out[agent_name] = self.critic_linear(x)
            else:
                env_critic_out[agent_name] = self.critic_linear[i](x)

            # mask logits of unavailable actions if provided
            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[agent_name][0, env_mask_idx[i]] = -1e10

        return env_actor_out, env_critic_out, hidden_state
