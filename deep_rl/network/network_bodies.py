#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import layer_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from .l0_layers import L0Dense

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


# class FCBody(nn.Module):
#     def __init__(self, state_dim, hidden_units=(64, 64), gate=torch.relu, dropout_rate=0,
#                  return_tc=False, tc_surrogate='no surrogate', lamba=1.0, weight_decay=1.0):
#         super(FCBody, self).__init__()
#         dims = (state_dim,) + hidden_units
#         if return_tc and tc_surrogate == 'l0':
#             droprate_init = 0.5
#             temperature = 2./3.
#             local_rep = False
#             self.N = 50000
#             self.weight_decay = weight_decay * self.N
#             if hidden_units.__len__() == 0:
#                 self.layers = []
#             else:
#                 self.layers = nn.ModuleList([layer_init(nn.Linear(state_dim, hidden_units[0]))] + [
#                     L0Dense(
#                         dim_in, dim_out, droprate_init=droprate_init, weight_decay=self.weight_decay,
#                         lamba=lamba, local_rep=local_rep, temperature=temperature
#                     ) for dim_in, dim_out in zip(dims[1:-1], dims[2:])
#                 ])
#         else:
#             self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
#         self.gate = gate
#         if dropout_rate != 0:
#             self.dropout_layer = nn.Dropout(dropout_rate)
#         else:
#             self.dropout_layer = None
#         self.return_tc = return_tc
#         self.tc_surrogate = tc_surrogate
#         self.feature_dim = dims[-1]
#
#     def forward(self, x):
#         if self.return_tc and self.tc_surrogate == 'l1_multiplication':
#             cost = 0
#             for layer in self.layers:
#                 tmp = x.unsqueeze(1) * layer.weight.unsqueeze(0)
#                 cost = cost + tmp.abs().sum(dim=1).sum(dim=1).unsqueeze(-1) + layer.bias.abs().sum()
#                 tmp = tmp.sum(dim=2) + layer.bias
#                 x = self.gate(tmp)
#                 if self.dropout_layer is not None:
#                     x = self.dropout_layer(x)
#             return x, cost
#         elif self.return_cost == 'sparse_neuron':
#             cost = 0
#             beta = 0.1
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 mask = (x.abs() > beta).type(torch.FloatTensor)
#                 tmp = (1. / (x.abs().detach() + 0.0001) - beta / (x.abs().detach() + 0.0001).pow(2)) * mask
#                 cost = cost + (x.abs() * tmp.detach()).sum(dim=1).unsqueeze(-1)
#                 if self.dropout_layer is not None:
#                     x = self.dropout_layer(x)
#             return x, cost
#         elif self.return_cost == 'l1_neuron':
#             cost = 0
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 cost = cost + x.abs().sum(dim=1).unsqueeze(-1)
#                 if self.dropout_layer is not None:
#                     x = self.dropout_layer(x)
#             return x, cost
#         elif self.return_cost == 'l2_neuron':
#             cost = 0
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 cost = cost + x.pow(2).sum(dim=1).unsqueeze(-1)
#                 if self.dropout_layer is not None:
#                     x = self.dropout_layer(x)
#             return x, cost
#         elif self.return_cost == 'l1_weight':
#             cost = 0
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 cost = cost + layer.weight.abs().sum().expand(x.shape[0], 1) + layer.bias.abs().sum()
#                 if self.dropout_layer is not None:
#                     x = self.dropout_layer(x)
#             return x, cost
#         elif self.return_cost == 'l2_weight':
#             cost = 0
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 cost = cost + layer.weight.pow(2).sum().expand(x.shape[0], 1) + layer.bias.pow(2).sum()
#                 if self.dropout_layer is not None:
#                     x = self.dropout_layer(x)
#             return x, cost
#         elif self.return_cost == 'l0':
#             cost = 0
#             is_first_layer = True
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 if is_first_layer is False:
#                     cost += - (1. / self.N) * layer.regularization().expand(x.shape[0], 1)
#                 is_first_layer = False
#             return x, cost, cost
#         elif self.return_cost == 'non_zero_weights':
#             cost = 0
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 cost = cost + (layer.weight.abs() > 0.001).sum().type(torch.FloatTensor).expand(x.shape[0], 1)
#                 if self.dropout_layer is not None:
#                     x = self.dropout_layer(x)
#             return x, cost
#         elif self.return_cost == 'non_zero_neurons':
#             cost = 0
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 cost = cost + (x != 0.0).sum(dim=1).type(torch.FloatTensor).unsqueeze(-1)
#                 if self.dropout_layer is not None:
#                     x = self.dropout_layer(x)
#             return x, cost
#         if self.return_tc is True:
#             tc = 0
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 tc = tc + layer.weight.shape[0] * layer.weight.shape[1] + layer.bias.shape[0]
#             return x, tc
#         else:
#             for layer in self.layers:
#                 x = layer(x)
#                 x = self.gate(x)
#                 if self.dropout_layer is not None:
#                     x = self.dropout_layer(x)
#             return x


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=torch.relu, return_tc=False):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.return_tc = return_tc
        self.feature_dim = dims[-1]

    def forward(self, x):
        if self.return_tc is True:
            tc = 0
            for layer in self.layers:
                x = layer(x)
                x = self.gate(x)
                # floating number multiplication/addition takes 1 tc unit
                # non-linear activation function on a floating number takes 1 tc unit
                tc = tc + layer.weight.shape[0] * layer.weight.shape[1] * 2 + layer.weight.shape[0]
            return x, tc
        else:
            for layer in self.layers:
                x = layer(x)
                x = self.gate(x)
            return x


class FCBodyWithTwoInputs(nn.Module):
    def __init__(
            self, input_dim1, input_dim2, combine='concatenate',
            hidden_units1=(64, 64), hidden_units2=(64, 64), gate1=torch.relu, gate2=torch.relu
    ):
        super(FCBodyWithTwoInputs, self).__init__()
        self.combine = combine
        if self.combine == 'first net only':
            self.n1 = FCBody(input_dim1, hidden_units1, gate1)
        elif self.combine == 'second net only':
            self.n2 = FCBody(input_dim2, hidden_units2, gate2)
        else:
            self.n1 = FCBody(input_dim1, hidden_units1, gate1)
            self.n2 = FCBody(input_dim2, hidden_units2, gate2)
        if combine in ['multiply', 'sum', 'subtract']:
            self.feature_dim = self.n1.feature_dim
        elif combine in ['first net only']:
            self.feature_dim = self.n1.feature_dim
        elif combine in ['second net only']:
            self.feature_dim = self.n2.feature_dim
        elif combine == 'concatenate':
            self.feature_dim = self.n1.feature_dim + self.n2.feature_dim
        elif combine == 'weighting':
            self.fc = layer_init(nn.Linear(self.n1.feature_dim, self.n1.feature_dim * self.n2.feature_dim))
            self.feature_dim = self.n1.feature_dim
        else:
            raise NotImplementedError

    def forward(self, input1, input2):
        if self.combine == 'first net only':
            return self.n1(input1)
        elif self.combine == 'second net only':
            return self.n2(input2)
        
        x = self.n1(input1)
        y = self.n2(input2)
        if self.combine == 'multiply':
            return x * y
        elif self.combine == 'sum':
            return x + y
        elif self.combine == 'subtract':
            return x - y
        elif self.combine == 'concatenate':
            return torch.cat((x, y), dim=1)
        elif self.combine == 'weighting':
            rtv = torch.tanh(torch.bmm(self.fc(x).view(-1, x.shape[1], y.shape[1]), y.unsqueeze(2)).view(-1, x.shape[1]))
            return rtv
        else:
            raise NotImplementedError
