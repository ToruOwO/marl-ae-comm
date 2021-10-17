from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.a3c_template import A3CTemplate, take_action, take_comm_action
from model.init import normalized_columns_initializer, weights_init
from model.model_utils import LSTMhead
from model.simclr import SimCLR

MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = {
    name: os.path.join(MODEL_DIR, f'../data/{name}_checkpoint_100.tar'
                       ) for name in ['resnet18', 'resnet50']
}


def load_simclr_encoder(proj_dim=64, encoder_name='resnet18'):
    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder_name=encoder_name, projection_dim=proj_dim)
    simclr_model.load_state_dict(torch.load(MODEL_PATH[encoder_name],
                                            map_location='cpu'))
    simclr_model.eval()
    return simclr_model.encoder, simclr_model.n_features


class PtSimCLR(nn.Module):
    def __init__(self, discrete_comm, comm_size=10, output_size=128,
                 ae_fc_size=0, aux_loss='', use_mlp=True,
                 encoder_name='resnet18'):
        super(PtSimCLR, self).__init__()
        self.simclr_encoder, self.n_features = load_simclr_encoder(
            encoder_name=encoder_name)
        self.discrete_comm = discrete_comm
        self.comm_size = comm_size
        self.output_size = output_size
        self.aux_loss = aux_loss

        if 'a' in aux_loss:
            # ae_fc_size <= 0: AE SimCLR
            # ae_fc_size > 0: fc / mlp + AE SimCLR
            self.ae = SimpleAE(self.n_features,
                               discrete_comm,
                               comm_size=comm_size,
                               ae_fc_size=ae_fc_size,
                               use_mlp=use_mlp)
        else:
            # fc / mlp + SimCLR
            self.enc = SimpleEncoder(self.n_features,
                                     discrete_comm,
                                     comm_size=comm_size,
                                     use_mlp=use_mlp)

            # similar to SimCLR projector
            self.feat_fc = nn.Sequential(
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, output_size),
            )

    def decode(self, x):
        if 'a' in self.aux_loss and self.ae.ae_fc_size <= 0:
            return self.ae.decoder(x).detach()
        else:
            return x

    def forward(self, x):
        """
        x - torch.Size([2, 3, 32, 32])

        return: in_feats, comm, comm_loss
        """
        # get simclr representation features
        with torch.no_grad():
            h = self.simclr_encoder(x)
        h = h.detach()  # (2, n_features)

        if 'a' in self.aux_loss:
            if self.ae.ae_fc_size > 0:
                # fc / mlp + AE SimCLR
                x_enc, z, loss = self.ae(h)
                return x_enc.detach(), z, loss
            else:
                # AE SimCLR
                _, z, loss = self.ae(h)
                return z.detach(), z, loss

        else:
            # fc / mlp + SimCLR
            z, _ = self.enc(h)  # encode h into comm
            y_out = self.feat_fc(h)
            return y_out, z, 0.0


class STEFunc(torch.autograd.Function):
    """Straight-Through Estimator"""

    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # clamp gradient between -1 and 1
        return F.hardtanh(grad_output)


class STE(nn.Module):
    def __init__(self):
        super(STE, self).__init__()

    def forward(self, x):
        x = STEFunc.apply(x)
        return x


class SimpleConv(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, input_size, discrete_comm, hidden_size=128, comm_size=1,
                 use_mlp=True):
        super(SimpleEncoder, self).__init__()

        if use_mlp:
            if discrete_comm:
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, comm_size),
                    nn.Sigmoid(),
                    STE()
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, comm_size),
                    nn.Sigmoid()
                )
        else:
            if discrete_comm:
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, comm_size),
                    nn.Sigmoid(),
                    STE()
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(input_size, comm_size),
                    nn.Sigmoid(),
                )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return z, x


class SimpleAE(nn.Module):
    def __init__(self, input_size, discrete_comm, hidden_size=128, comm_size=1,
                 ae_fc_size=0, use_mlp=True):
        super(SimpleAE, self).__init__()
        self.ae_fc_size = ae_fc_size

        if ae_fc_size > 0:
            # fc / mlp + AE SimCLR
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, ae_fc_size),
            )
            self.decoder = nn.Sequential(
                nn.Linear(ae_fc_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
            )
            self.fc = SimpleFC(ae_fc_size, discrete_comm, comm_size,
                               use_mlp)
        else:
            # AE SimCLR
            self.encoder = SimpleEncoder(input_size, discrete_comm, hidden_size,
                                         comm_size, use_mlp=True)
            self.decoder = nn.Sequential(
                nn.Linear(comm_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
            )
            self.fc = None

    def forward(self, x):
        if self.ae_fc_size > 0:
            # fc / mlp + AE SimCLR

            # z - (2, comm_size)
            # x - (2, n_features)
            # x_enc - (2, ae_fc_size)
            x = x.view(x.size(0), -1)
            x_enc = self.encoder(x)
            x_out = self.decoder(x_enc)

            # do not update AE with policy gradient
            z = self.fc(x_enc.detach())

            loss = F.mse_loss(x_out, x)
            return x_enc, z, loss
        else:
            # AE SimCLR

            # z - (2, comm_size)
            # x - (2, n_features)
            z, x = self.encoder(x)
            x_out = self.decoder(z)
            loss = F.mse_loss(x_out, x)
            return x, z, loss


class SimpleFC(nn.Module):
    def __init__(self, input_size, discrete_comm, comm_size, use_mlp=False):
        super(SimpleFC, self).__init__()
        if use_mlp:
            hidden_size = 128
            self.base = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, comm_size),
            )
        else:
            self.base = nn.Linear(input_size, comm_size)

        if discrete_comm:
            self.head = nn.Sequential(
                nn.Sigmoid(),
                STE()
            )
        else:
            self.head = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)
        return self.head(x)


class CifarNet(A3CTemplate):
    """
    A network with AE comm.
    """

    def __init__(self, act_space, comm_type, comm_pg, aux_loss, img_feat_size,
                 hidden_size=128, comm_size=1, ae_fc_size=0, use_mlp=False,
                 debug=False, simclr_encoder_name='resnet18'):
        super().__init__()
        self.debug = debug

        # assume action space is a Tuple of 2 spaces
        self.env_action_size = act_space[0].n
        self.comm_action_space = act_space[1]

        # check whether comm action is discrete or continuous
        if self.comm_action_space.__class__.__name__ == 'MultiBinary':
            self.discrete_comm = True
        elif self.comm_action_space.__class__.__name__ == 'Box':
            self.discrete_comm = False
        else:
            raise NotImplementedError

        self.action_size = self.env_action_size
        self.comm_size = comm_size
        self.num_agents = 2

        if comm_type == 1:
            self.net = PtSimCLR(discrete_comm=self.discrete_comm,
                                comm_size=comm_size,
                                output_size=img_feat_size,
                                ae_fc_size=ae_fc_size,
                                aux_loss=aux_loss,
                                use_mlp=use_mlp,
                                encoder_name=simclr_encoder_name)
            if 'a' in aux_loss:
                if ae_fc_size > 0:
                    # additional fc on top of the autoencoded simclr rep
                    img_feat_size = ae_fc_size
                else:
                    img_feat_size = self.net.n_features
        else:
            self.net = SimpleConv(output_size=img_feat_size)

        # individual memories
        if comm_type == 0:
            self.feat_dim = img_feat_size  # img feats
        elif comm_type == 1:
            # img feats + comm obs + self comm
            self.feat_dim = img_feat_size + comm_size * 2
        elif comm_type == 2:
            # img feats + comm obs + last self comm
            self.feat_dim = img_feat_size + comm_size * 2
            self.comm_critic_linear = nn.ModuleList([nn.Linear(
                hidden_size, 1) for _ in range(self.num_agents)])

            if self.discrete_comm:
                self.comm_actor_linear = nn.ModuleList([nn.Linear(
                    hidden_size, 2 * comm_size
                ) for _ in range(self.num_agents)])
            else:
                self.comm_actor_linear = nn.ModuleList([nn.Sequential(
                    nn.Linear(hidden_size, comm_size),
                    nn.Sigmoid()
                ) for _ in range(self.num_agents)])
        elif comm_type == 3:
            # img feats + comm obs + last self comm
            self.feat_dim = img_feat_size + comm_size * 2
            self.comm_net = SimpleFC(img_feat_size,
                                     discrete_comm=self.discrete_comm,
                                     comm_size=comm_size,
                                     use_mlp=use_mlp)
        else:
            raise ValueError(f'comm_type {comm_type} not supported')

        self.head = nn.ModuleList(
            [LSTMhead(self.feat_dim, hidden_size) for _ in range(
                self.num_agents)])
        self.is_recurrent = True

        # separate AC for env action and comm action
        self.env_critic_linear = nn.ModuleList([nn.Linear(
            hidden_size, 1) for _ in range(self.num_agents)])
        self.env_actor_linear = nn.ModuleList([nn.Linear(
            hidden_size, self.env_action_size) for _ in range(self.num_agents)])

        self.comm_type = comm_type
        self.comm_pg = comm_pg
        self.aux_loss = aux_loss

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.env_actor_linear:
            m.weight.data = normalized_columns_initializer(
                m.weight.data, 0.01)
            m.bias.data.fill_(0)

        for m in self.env_critic_linear:
            m.weight.data = normalized_columns_initializer(
                m.weight.data, 1.0)
            m.bias.data.fill_(0)

    def init_hidden(self):
        return [head.init_hidden() for head in self.head]

    def take_comm_action(self, comm_out):
        if self.discrete_comm:
            if self.comm_size == 1:
                comm_act = int(comm_out[0].item())
            else:
                comm_act = [int(comm_out[i].item()
                                ) for i in range(len(comm_out))]
        else:
            if self.comm_size == 1:
                comm_act = comm_out[0].item()
            else:
                comm_act = [comm_out[i].item() for i in range(len(comm_out))]
        return comm_act

    def take_action(self, policy_logit, comm_out):
        act_list = []
        act_logp_list = []
        ent_list = []
        all_act = []

        if self.comm_type == 2:
            # use comm as an RL action
            env_logits, comm_logits = policy_logit
            for i in range(self.num_agents):
                envl = env_logits[i]
                comml = comm_logits[i]
                env_act, env_act_logp, env_ent = take_action(
                    envl, self.env_action_size)
                comm_act, comm_act_logp, comm_ent = take_comm_action(
                    comml, self.comm_action_space)

                act_list.append([env_act, comm_act])
                act_logp_list.append([env_act_logp, comm_act_logp])

                if self.discrete_comm:
                    ent_list.append([env_ent, *comm_ent])
                else:
                    ent_list.append([env_ent, comm_ent])
                all_act.append([env_act, comm_act])

            act_list = list(zip(*act_list))
            act_logp_list = list(zip(*act_logp_list))
        else:
            # use comm for message passing only
            for i in range(self.num_agents):
                logits = policy_logit[i]
                act, act_logp, ent = super(CifarNet, self).take_action(logits)

                act_list.append(act)
                act_logp_list.append(act_logp)
                ent_list.append(ent)

                comm_act = self.take_comm_action(comm_out[i])

                all_act.append([act, comm_act])

        return act_list, act_logp_list, ent_list, all_act

    def forward(self, inputs, hidden_state=None, env_mask_idx=None):
        assert len(inputs) == 2
        assert type(inputs[0]) is dict

        # WARNING: the following code only works for Python 3.6 and beyond

        if self.debug:
            print('[MODEL] aux loss', self.aux_loss)

        # (1) pre-process inputs
        imgs = torch.stack([x['img'] for x in inputs], dim=0)  # (2, C, H, W)
        if self.comm_type == 1:
            # in_feats - (2, img_feat_size)
            # comm - (2, comm_size)
            in_feats, comm, comm_loss = self.net(imgs)

            if self.debug:
                print('[MODEL] img feats', in_feats.shape)
                print(f'[MODEL] {self.aux_loss} comm loss', comm_loss)
        elif self.comm_type == 3:
            # fc / mlp on pixel
            in_feats = self.net(imgs)  # (2, img_feat_size)
            comm = self.comm_net(in_feats)
            comm_loss = 0.0

            if self.debug:
                print('[MODEL] img feats', in_feats.shape)
        else:
            in_feats = self.net(imgs)  # (2, img_feat_size)
            comm_loss = 0.0

        # (2) predict policy and values separately
        env_actor_out, env_critic_out = {}, {}
        comm_actor_out, comm_critic_out = {}, {}
        comm_out = []

        for i in range(self.num_agents):

            # get comm

            if self.comm_type == 0:
                # no comm
                feats = in_feats[i:i + 1]
                c = torch.zeros(1, 1)
                comm_out.append(c)
                x, hidden_state[i] = self.head[i](feats, hidden_state[i])

            elif self.comm_type == 2:
                # rl comm
                feats = torch.cat([in_feats[i],
                                   inputs[i]['comm'],
                                   inputs[i]['selfcomm']],
                                  dim=-1).unsqueeze(0)
                x, hidden_state[i] = self.head[i](feats, hidden_state[i])

                comm_actor_out[i] = self.comm_actor_linear[i](x)
                comm_critic_out[i] = self.comm_critic_linear[i](x)

            elif self.comm_type == 3:
                comm_out.append(comm[i].detach())
                if self.comm_pg:
                    feats = torch.cat([in_feats[i],
                                       inputs[i]['comm'],
                                       comm[i]],
                                      dim=-1).unsqueeze(0)
                else:
                    feats = torch.cat([in_feats[i],
                                       inputs[i]['comm'],
                                       comm_out[i]],  # detached comm
                                      dim=-1).unsqueeze(0)
                x, hidden_state[i] = self.head[i](feats, hidden_state[i])

            else:
                # non-rl comm
                comm_out.append(comm[i].detach())
                other_comm = self.net.decode(inputs[i]['comm'])

                if self.comm_pg:
                    feats = torch.cat([in_feats[i],
                                       other_comm,
                                       comm[i]],
                                      dim=-1).unsqueeze(0)
                else:
                    feats = torch.cat([in_feats[i],
                                       other_comm,
                                       comm_out[i]],  # detached comm
                                      dim=-1).unsqueeze(0)
                x, hidden_state[i] = self.head[i](feats, hidden_state[i])

            if self.debug:
                print('[MODEL] feats', feats.shape)
                print('[MODEL] comm', comm.shape, comm[0])
                assert False

            # get env policy logits and predicted values
            env_actor_out[i] = self.env_actor_linear[i](x)
            env_critic_out[i] = self.env_critic_linear[i](x)

            # mask logits of unavailable actions if provided
            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[i][0, env_mask_idx[i]] = -1e10

        if self.comm_type == 2:
            return (env_actor_out, comm_actor_out), \
                   (env_critic_out, comm_critic_out), \
                   hidden_state, comm_out, comm_loss
        else:
            return env_actor_out, env_critic_out, hidden_state, comm_out, \
                   comm_loss
