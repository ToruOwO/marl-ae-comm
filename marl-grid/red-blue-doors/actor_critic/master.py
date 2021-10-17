from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import time, os, sys
from collections import deque
from loss import policy_gradient_loss
from torch.utils.tensorboard import SummaryWriter


def set_requires_grad(modules, value):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = value


class Master(object):
    """ A master network. Think of it as a container that holds weight and the
    optimizer for the workers

    Args
        net: a neural network A3C model
        opt: shared optimizer
        gpu_id: gpu device id
    """

    def __init__(self, net, opt, global_iter, global_done, master_lock,
                 writer_dir, max_iteration=100):
        self.lock = master_lock
        self.iter = global_iter
        self.done = global_done
        self.max_iteration = max_iteration
        self.net = net
        self.opt = opt
        self.net.share_memory()
        self.writer_dir = writer_dir

    def init_tensorboard(self):
        """ initializes tensorboard by the first worker """
        with self.lock:
            if not hasattr(self, 'writer'):
                self.writer = SummaryWriter(self.writer_dir)
        return

    def copy_weights(self, net, with_lock=False):
        """ copy weight from master """

        if with_lock:
            with self.lock:
                for p, mp in zip(net.parameters(), self.net.parameters()):
                    p.data.copy_(mp.data)
            return self.iter.value
        else:
            for p, mp in zip(net.parameters(), self.net.parameters()):
                p.data.copy_(mp.data)
            return self.iter.value

    def _apply_gradients(self, net):
        # backward prop and clip gradients
        self.opt.zero_grad()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 40.0)

        for p, mp in zip(net.parameters(), self.net.parameters()):
            if p.grad is not None:
                mp.grad = p.grad.cpu()
        self.opt.step()

    def apply_gradients(self, net, with_lock=False):
        """ apply gradient to the master network """
        if with_lock:
            with self.lock:
                self._apply_gradients(net)
        else:
            self._apply_gradients(net)
        return

    def increment(self, progress_str=None):
        with self.iter.get_lock():
            self.iter.value += 1

            if self.iter.value % 100 == 0:

                if progress_str is not None:
                    print('[{}/{}] {}'.format(
                        self.iter.value, self.max_iteration, progress_str))

                else:
                    print('[{}/{}] workers are working hard.'.format(
                        self.iter.value, self.max_iteration))

            if self.iter.value > self.max_iteration:
                self.done.value = 1
        return

    def is_done(self):
        return self.done.value

    def save_ckpt(self, weight_iter, save_path):
        torch.save({'net': self.net.state_dict(),
                    'opt': self.opt.state_dict(),
                    'iter': weight_iter}, save_path)
