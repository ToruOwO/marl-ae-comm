import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import ops


def take_action(policy_logit, action_size):
    """ for discrete or multi-discrete policy, returns argmax as a numpy """
    assert policy_logit.size(0) == 1
    assert len(policy_logit.size()) == 2

    policy = F.softmax(policy_logit[0], dim=0)
    log_policy = F.log_softmax(policy_logit[0], dim=0)
    entropy = -(policy * log_policy).sum()

    policy = ops.to_numpy(policy)
    action = np.random.choice(action_size, p=policy)
    action_logp = log_policy[action]
    return action, action_logp, entropy


def take_comm_action(policy_logit, action_space):
    """for box / discrete / multi-discrete policy"""
    assert policy_logit.size(0) == 1
    assert len(policy_logit.size()) == 2

    if action_space.__class__.__name__ == 'MultiDiscrete':
        action_sizes = action_space.nvec.tolist()
        policy_logit = torch.split(policy_logit, action_sizes, dim=-1)
        action_info = [take_action(logit, action_sizes[i]
                                   ) for i, logit in enumerate(policy_logit)]
        action, action_logp, entropy = list(zip(*action_info))
        return np.asarray(action), action_logp, entropy
    elif action_space.__class__.__name__ == 'Discrete':
        return take_action(policy_logit, action_space.n)
    elif action_space.__class__.__name__ == 'Box':
        if action_space.shape[0] == 1:
            return policy_logit[0].item(), 0., 0.
        else:
            return [policy_logit[0][i].item() for i in range(
                policy_logit.shape[-1])], 0., 0.
    else:
        raise NotImplementedError


class A3CTemplate(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def take_action(self, policy_logit):
        """ for discrete policy, returns argmax as a numpy """
        return take_action(policy_logit, self.action_size)

    def init_hidden(self):
        """ initializes zero state. Has dim (2 x num_layers x 1 x feat_dim) """
        assert self.head.is_recurrent, 'policy head is not recurrent'
        return self.head.init_hidden()
