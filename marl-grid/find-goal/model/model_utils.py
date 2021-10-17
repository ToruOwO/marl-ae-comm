from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.init import weights_init


class LSTMhead(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=1):
        super().__init__()
        self.is_recurrent = True
        self.lstm = nn.LSTM(in_dim, out_dim, num_layers=num_layers)
        self.hidden_shape = (2, 1, 1, 256)
        self.reset_parameters()
        return

    def reset_parameters(self):
        self.lstm.weight_ih_l0.data.fill_(0)
        self.lstm.weight_hh_l0.data.fill_(0)
        return

    def forward(self, x, hidden_state):
        x = x.unsqueeze(0)
        x, hidden_state = self.lstm(x, hidden_state)
        return x[0], hidden_state

    def init_hidden(self):
        """ initializes zero state (2 x num_layers x 1 x feat_dim) """
        assert self.is_recurrent, 'model is not recurrent'
        return (torch.zeros(1, 1, 256).cuda(), torch.zeros(1, 1, 256).cuda())


class ImgModule(nn.Module):
    """Process image inputs of shape CxHxW."""
    def __init__(self, input_size, last_fc_dim=0):
        super(ImgModule, self).__init__()
        self.conv1 = self.make_layer(input_size[2], 32)
        self.conv2 = self.make_layer(32, 32)
        self.conv3 = self.make_layer(32, 32)
        self.conv4 = self.make_layer(32, 32)
        self.avgpool = nn.AdaptiveAvgPool2d([3, 3])
        if last_fc_dim > 0:
            self.fc = nn.Linear(288, last_fc_dim)
        else:
            self.fc = None
        self.apply(weights_init)

    def make_layer(self, in_ch, out_ch, use_norm=True):
        layer = []
        layer += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)]
        layer += [nn.ELU(True)]
        if use_norm:
            layer += [nn.InstanceNorm2d(out_ch)]
        return nn.Sequential(*layer)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(-1, 32 * 3 * 3)  # feature dim
        if self.fc is not None:
            return self.fc(x)
        return x


class CommModule(nn.Module):
    """Process discrete / continuous binary messages of shape
    (B, num_agents, comm_len)."""
    def __init__(self, comm_size, comm_len, discrete_comm, hidden_size,
                 emb_size, comm_rnn):
        super(CommModule, self).__init__()
        assert comm_size == 2

        if discrete_comm:
            self.emb = nn.Embedding(comm_size, emb_size)
            self.emb_fc = nn.Sequential(nn.Linear(emb_size, 32), nn.ReLU(),
                                        nn.Linear(32, 32), nn.ReLU())
        else:
            self.fc = nn.Sequential(nn.Linear(comm_len, emb_size), nn.ReLU(),
                                    nn.Linear(emb_size, 32), nn.ReLU())

        if comm_rnn:
            self.rnn = nn.GRU(32, hidden_size, batch_first=True)

        self.comm_rnn = comm_rnn
        self.comm_len = comm_len
        self.discrete_comm = discrete_comm

    def forward(self, inputs):
        B = inputs.shape[0]
        N = inputs.shape[-2]  # num_agents

        # get embedded communication
        if self.discrete_comm:
            emb = self.emb(inputs.long())  # (B, N, comm_len, emb_size)
            emb = self.emb_fc(emb)  # (B, N, comm_len, 32)

            emb = emb.view(-1, N * self.comm_len, 32)  # (B, N*comm_len, 32)
        else:
            emb = self.fc(inputs)  # (B, N, 32)

        if self.comm_rnn:
            _, hidden = self.rnn(emb)  # (1, B, hidden_size)
            return hidden[-1]
        else:
            return emb.view(B, -1)


class InputProcessingModule(nn.Module):
    """
    Pre-process the following individual observations:
        - pov (ImgModule)
        - t
        - self_env_act
        - selfpos
        - comm (CommModule)
        - position
        - done
    """
    def __init__(self, obs_space, comm_size, comm_len, discrete_comm, emb_size,
                 num_agents, num_blind_agents, layer_norm, comm_rnn=True,
                 num_adversaries=0):
        super(InputProcessingModule, self).__init__()

        self.obs_keys = list(obs_space.spaces.keys())

        self.num_agents = num_agents
        self.num_blind_agents = num_blind_agents
        self.num_adversaries = num_adversaries

        # image processor
        if 'pov' in self.obs_keys:
            self.conv = ImgModule(obs_space['pov'].shape)
            feat_dim = 32 * 3 * 3
        else:
            feat_dim = 0

        # state inputs processor
        state_feat_dim = 0

        # global / private states
        if 't' in self.obs_keys:
            state_feat_dim += 1

        if 'self_env_act' in self.obs_keys:
            # discrete value with one-hot encoding
            self.env_act_dim = obs_space.spaces['self_env_act'].n
            state_feat_dim += self.env_act_dim

        if 'selfpos' in self.obs_keys:
            self.discrete_positions = None
            if obs_space.spaces['selfpos'].__class__.__name__ == \
                    'MultiDiscrete':
                # process position with one-hot encoder
                self.discrete_positions = obs_space.spaces['selfpos'].nvec
                state_feat_dim += sum(self.discrete_positions)
            else:
                state_feat_dim += 2

        # states that contain other agents' specific information
        if 'comm' in self.obs_keys:
            self.comm = CommModule(comm_size, comm_len, discrete_comm,
                                   64, emb_size, comm_rnn)
            if comm_rnn:
                state_feat_dim += 64
            else:
                if discrete_comm:
                    state_feat_dim += num_agents * comm_len * 32
                else:
                    state_feat_dim += num_agents * 32

        if 'position' in self.obs_keys:
            self.discrete_positions = None
            if obs_space.spaces['position'].__class__.__name__ == \
                    'Tuple':
                assert len(obs_space.spaces['position']) == num_agents
                assert obs_space.spaces['position'][0].__class__.__name__ == \
                    'MultiDiscrete'

                # process position with one-hot encoder
                self.discrete_positions = obs_space.spaces['position'][0].nvec
                state_feat_dim += num_agents * sum(self.discrete_positions)
            else:
                state_feat_dim += num_agents * 2

        if 'done' in self.obs_keys:
            state_feat_dim += num_agents

        if state_feat_dim == 0:
            self.state_feat_fc = None
        else:
            # use state_feat_fc to process concatenated state inputs
            self.state_feat_fc = nn.Linear(state_feat_dim, 64)
            feat_dim += 64

        self.feat_dim = feat_dim

        self.layer_norm = layer_norm
        if layer_norm:
            self.state_layer_norm = nn.LayerNorm(64)
            self.img_layer_norm = nn.LayerNorm(288)

    def forward(self, inputs):
        # WARNING: the following code only works for Python 3.6 and beyond

        # process images together if provided
        # (3, view_size * view_tile_size, view_size * view_tile_size)
        if 'pov' in self.obs_keys:
            pov = []
            for i in range(self.num_blind_agents, self.num_agents):
                pov.append(inputs[f'agent_{i}']['pov'])
            x = torch.cat(pov, dim=0)
            x = self.conv(x)  # (N - N_blind, img_feat_dim)
            xs = torch.chunk(x, self.num_agents - self.num_blind_agents)

        # process communication
        if 'comm' in self.obs_keys:
            c = []
            for i in range(self.num_agents):
                c.append(inputs[f'agent_{i}']['comm'])
            c = torch.stack(c, dim=0)
            c = self.comm(c)

        # process (normalized) time if provided
        if 't' in self.obs_keys:
            t = torch.zeros(1, 1).cuda()
            t[0] = inputs[f'agent_{i}']['t']
        else:
            t = None

        if self.state_feat_fc is None:
            img_feat = [torch.zeros(1, 288).cuda() for _ in range(
                self.num_blind_agents)] + [self.img_layer_norm(f) for f in xs]
            return img_feat

        # concatenate observation features
        cat_feat = [None for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            feats = []

            # concat (normalized) time if provided
            if t is not None:
                feats.append(t)

            # concat last env act if provided
            if 'self_env_act' in self.obs_keys:
                env_act = F.one_hot(
                    inputs[f'agent_{i}']['self_env_act'].to(torch.int64),
                    num_classes=self.env_act_dim)
                env_act = torch.reshape(env_act, (1, self.env_act_dim))
                feats.append(env_act)

            # concat agent's own position if provided
            if 'selfpos' in self.obs_keys:
                sp = inputs[f'agent_{i}']['selfpos'].to(torch.int64)  # (2,)
                if self.discrete_positions is not None:
                    spx = F.one_hot(sp[0],
                                    num_classes=self.discrete_positions[0])
                    spy = F.one_hot(sp[1],
                                    num_classes=self.discrete_positions[1])
                    sp = torch.cat([spx, spy], dim=-1).float()
                    sp = torch.reshape(sp, (1, sum(self.discrete_positions)))
                else:
                    sp = torch.reshape(sp, (1, 2))
                feats.append(sp)

            # concat comm features for each agent if provided
            if 'comm' in self.obs_keys:
                feats.append(c[i:i+1])

            if 'position' in self.obs_keys and 'done' in self.obs_keys:
                # position
                p = inputs[f'agent_{i}']['position'].to(torch.int64)
                if self.discrete_positions is not None:
                    px = F.one_hot(p[:, 0],
                                   num_classes=self.discrete_positions[0])
                    py = F.one_hot(p[:, 1],
                                   num_classes=self.discrete_positions[1])
                    p = torch.cat([px, py], dim=-1).squeeze(-2)

                # done
                d = torch.reshape(inputs[f'agent_{i}']['done'],
                                  (self.num_agents, 1))

                pd = torch.cat([p, d], dim=-1)  # (num_agent, pos_dim * 2 + 1)
                pd = torch.reshape(pd, (1, -1))

                feats.append(pd)

            else:
                if 'position' in self.obs_keys:
                    p = inputs[f'agent_{i}']['position'].to(torch.int64)
                    if self.discrete_positions is not None:
                        px = F.one_hot(p[:, 0],
                                       num_classes=self.discrete_positions[0])
                        py = F.one_hot(p[:, 1],
                                       num_classes=self.discrete_positions[1])
                        p = torch.cat([px, py], dim=-1).squeeze(-2)
                        p = torch.reshape(p, (1, self.num_agents * sum(
                            self.discrete_positions)))
                    else:
                        p = torch.reshape(p, (1, self.num_agents * 2))
                    feats.append(p)

                if 'done' in self.obs_keys:
                    d = torch.reshape(inputs[f'agent_{i}']['done'],
                                      (1, self.num_agents))
                    feats.append(d)

            if len(feats) > 1:
                cat_feat[i] = torch.cat(feats, dim=-1)
            else:
                cat_feat[i] = feats[0]
            cat_feat[i] = self.state_feat_fc(cat_feat[i])
            if self.layer_norm:
                cat_feat[i] = self.state_layer_norm(cat_feat[i])

            if 'pov' in self.obs_keys:
                if i >= self.num_blind_agents:
                    if self.layer_norm:
                        img_feat = self.img_layer_norm(
                            xs[i - self.num_blind_agents])
                    else:
                        img_feat = xs[i - self.num_blind_agents]

                    cat_feat[i] = torch.cat([cat_feat[i], img_feat], dim=-1)

        return cat_feat
