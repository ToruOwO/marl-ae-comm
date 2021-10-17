from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F
import numpy as np


def kld_loss(mu, var):
    bsz = mu.size()[0]
    mu, var = mu.contiguous(), var.contiguous()
    mu, var = mu.view(bsz, -1), var.view(bsz, -1)
    return torch.mean(0.5 * (mu ** 2 + torch.exp(var) - var - 1), dim=1)


def reparamterize(mu_var, use_gpu=False):
    vae_dim = int(mu_var.size()[1] / 2)
    mu, var = mu_var[:, :vae_dim], mu_var[:, vae_dim:]
    eps = to_torch(torch.randn(mu.size()), use_gpu=use_gpu)
    z = mu + eps * torch.exp(var / 2)  # var -> std
    return z, mu, var


def discrete_policy_gradient_loss(policy_logit, action, advantage, gae=False,
                                  value_weight=0.5, entropy_weight=0.01):
    policy = F.softmax(policy_logit, dim=-1)[0]
    log_policy = F.log_softmax(policy_logit, dim=-1)[0]
    log_policy_action = log_policy[action]

    if gae is not False and gae is not None:
        policy_loss = -log_policy_action * gae[0].detach()
    else:
        policy_loss = -log_policy_action * advantage[0]

    value_loss = advantage ** 2
    entropy = - (policy * log_policy).sum()

    loss = policy_loss + \
           value_weight * value_loss - \
           entropy_weight * entropy

    return loss, policy_loss, value_loss, entropy


def policy_gradient_loss(policy_logit, action, advantage, gae=False,
                         value_weight=0.5, entropy_weight=0.01,
                         action_space=None):
    if action_space is None:
        # default to Discrete
        loss, policy_loss, value_loss, entropy = discrete_policy_gradient_loss(
            policy_logit, action, advantage, gae, value_weight, entropy_weight
        )
        return loss, (policy_loss, value_loss, entropy)

    elif action_space.__class__.__name__ == 'MultiBinary':
        # treat them as independent discrete actions
        policy_logit = torch.split(policy_logit,
                                   [2 for _ in range(action_space.n)],
                                   dim=-1)

        loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0
        for i, logit in enumerate(policy_logit):
            l, pl, vl, ent = discrete_policy_gradient_loss(
                logit, action[i], advantage, gae, value_weight, entropy_weight)
            loss += l
            policy_loss += pl
            value_loss += vl
            entropy += ent
        loss /= len(action)
        policy_loss /= len(action)
        value_loss /= len(action)
        entropy /= len(action)

        return loss, (policy_loss, value_loss, entropy)

    elif action_space.__class__.__name__ == 'Discrete':
        loss, policy_loss, value_loss, entropy = discrete_policy_gradient_loss(
            policy_logit, action, advantage, gae, value_weight, entropy_weight
        )
        return loss, (policy_loss, value_loss, entropy)

    elif action_space.__class__.__name__ == 'MultiDiscrete':
        # treat them as independent discrete actions
        policy_logit = torch.split(policy_logit, action_space.nvec.tolist(),
                                   dim=-1)

        loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0
        for i, logit in enumerate(policy_logit):
            l, pl, vl, ent = discrete_policy_gradient_loss(
                logit, action[i], advantage, gae, value_weight, entropy_weight)
            loss += l
            policy_loss += pl
            value_loss += vl
            entropy += ent
        loss /= len(action)
        policy_loss /= len(action)
        value_loss /= len(action)
        entropy /= len(action)

        return loss, (policy_loss, value_loss, entropy)

    elif action_space.__class__.__name__ == 'Box':
        value_loss = advantage ** 2
        return value_loss, (0., value_loss, 0.)

    else:
        raise NotImplementedError
