from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np


def to_torch(x, use_gpu=True, dtype=np.float32):
    x = np.array(x, dtype=dtype)
    var = torch.from_numpy(x)
    return var.cuda() if use_gpu is not None else var


def to_numpy(x):
    if isinstance(x, int) or isinstance(x, float):
        return x
    if isinstance(x, (list, np.ndarray)):
        return np.array([to_numpy(_x) for _x in x])
    return x.detach().cpu().numpy()


def norm_col_init(weights, std=1.0):
    """
    Normalized column initializer
    """
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x
