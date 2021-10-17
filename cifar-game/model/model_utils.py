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
        self.hidden_size = out_dim
        self.hidden_shape = (2, 1, 1, out_dim)
        self.reset_parameters()
        return

    def reset_parameters(self):
        return

    def forward(self, x, hidden_state):
        x = x.unsqueeze(0)
        x, hidden_state = self.lstm(x, hidden_state)
        return x[0], hidden_state

    def init_hidden(self):
        """ initializes zero state (2 x num_layers x 1 x feat_dim) """
        assert self.is_recurrent, 'model is not recurrent'
        return (torch.zeros(1, 1, self.hidden_size).cuda(),
                torch.zeros(1, 1, self.hidden_size).cuda())
