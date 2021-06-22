#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch.nn as nn
import numpy as np
import torch


class BaseNet:
    def __init__(self):
        pass


# def layer_init(layer, w_scale=1):
#     nn.init.orthogonal_(layer.weight.data)
#     # nn.init.constant_(layer.weight.data, 10)
#     # nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2))
#     layer.weight.data.mul_(w_scale)
#     # nn.init.constant_(layer.bias.data, 0)
#     # nn.init.normal_(layer.bias.data)
#     if layer.bias is not None:
#         nn.init.constant_(layer.bias.data, 0)
#
#     return layer

# def layer_init(layer, w_scale=1.0):
#     nn.init.orthogonal_(layer.weight.data)
#     # nn.init.constant_(layer.weight.data, 10)
#     # nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2))
#     # layer.weight.data.mul_(w_scale)
#     # nn.init.constant_(layer.bias.data, 0)
#     # nn.init.normal_(layer.bias.data)
#     if layer.bias is not None:
#         nn.init.constant_(layer.bias.data, 0)
#
#     return layer


def layer_init(layer, w_scale=1.0, initialization='kaiming_uniform'):
    if initialization == 'kaiming_uniform':
        # torch.manual_seed(0)
        nn.init.kaiming_uniform_(layer.weight.data, mode='fan_in', nonlinearity='tanh')
        # layer.weight.data.mul_(10)
    elif initialization == 'all_zeros':
        nn.init.constant_(layer.weight.data, 0)
    elif initialization == 'all_ones':
        nn.init.constant_(layer.weight.data, 1)
    else:
        raise NotImplementedError
    if layer.bias is not None:
        nn.init.constant_(layer.bias.data, 0)
    return layer