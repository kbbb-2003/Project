import numpy as np
import torch.nn as nn
from algs.utils.norm import RunningMeanStd


def orth_init(layer, gain=1.0):
    # trick 8: orth initialization
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape: 1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # only divided std

        return x

    def reset(self):  # when an episode is done, reset self.R
        self.R = np.zeros(self.shape)
