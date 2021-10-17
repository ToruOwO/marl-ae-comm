"""Adapted from https://github.com/Spijkervet/SimCLR"""

import torch.nn as nn
import torchvision

from model.simclr_modules.resnet_hacks import modify_resnet_model
from model.simclr_modules.identity import Identity


def get_resnet(name, pretrained=False):
    resnets = {
        'resnet18': torchvision.models.resnet18(pretrained=pretrained),
        'resnet50': torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f'{name} is not a valid ResNet version')
    return resnets[name]


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016)
    to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the
    average pooling layer.
    """

    def __init__(self, encoder_name='resnet18', projection_dim=64):
        super(SimCLR, self).__init__()

        self.encoder = get_resnet(encoder_name, pretrained=False)

        # get dimensions of last fully-connected layer of encoder
        # (2048 for resnet50, 512 for resnet18)
        self.n_features = self.encoder.fc.in_features

        # replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # use a MLP with one hidden layer to obtain
        # z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
