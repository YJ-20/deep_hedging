#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class BaseNet:
    def __init__(self):
        pass


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def rnn_init(layer, initializer):
    if initializer == 'xavier':
        nn.init.xavier_normal_(layer.weight_hh_l0.data, gain=1.0)
        nn.init.xavier_normal_(layer.weight_ih_l0.data, gain=1.0)
    elif initializer == 'he':
        nn.init.kaiming_normal_(layer.weight_hh_l0.data)
        nn.init.kaiming_normal_(layer.weight_ih_l0.data)
    nn.init.constant_(layer.bias_hh_l0.data, 0)
    nn.init.constant_(layer.bias_ih_l0.data, 0)
    return layer
