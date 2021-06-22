#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_utils import BaseNet, layer_init
from ..utils.torch_utils import tensor, to_np
from ..utils.param_config import ParamConfig
from .network_bodies import FCBody, FCBodyWithTwoInputs
from .l0_layers import L0Dense
import itertools

class LinearNet(nn.Module, BaseNet):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearNet, self).__init__()
        # self.w = layer_init(nn.Linear(input_dim, output_dim, bias=False), w_scale=0.01)
        self.w = layer_init(nn.Linear(input_dim, output_dim, bias=bias))
        # self.w = layer_init_xavier(nn.Linear(input_dim, output_dim, bias=False))
        self.to(ParamConfig.DEVICE)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        y = self.w(tensor(x))
        return y


class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body, bias=True, initialization="kaiming_uniform"):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim, bias=bias), initialization=initialization)
        self.body = body
        self.to(ParamConfig.DEVICE)
        pass

    def forward(self, x):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        return y
    

class OptionValueLearningNet(nn.Module, BaseNet):
    def __init__(self, learn_length, output_dim, body, bias=True, initialization="kaiming_uniform"):
        super(OptionValueLearningNet, self).__init__()
        self.fc_option_values = layer_init(
            nn.Linear(body.feature_dim, output_dim, bias=bias), initialization=initialization
        )
        self.learn_length = learn_length
        if learn_length:
            self.fc_option_lengths = layer_init(
                nn.Linear(body.feature_dim, output_dim, bias=bias),
                initialization='all_ones'
            )
        self.reward_rate = layer_init(nn.Linear(1, 1, bias=False), initialization=initialization)
        self.body = body
        self.to(ParamConfig.DEVICE)
        pass

    def forward(self, x):
        phi = self.body(tensor(x))
        option_values = self.fc_option_values(phi)
        reward_rate = self.reward_rate(tensor([[1.]]))
        if self.learn_length:
            option_lengths = self.fc_option_lengths(phi)
            return option_values, option_lengths, reward_rate
        else:
            return option_values, reward_rate


class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, bias=True):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1, bias=bias))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim, bias=bias))
        self.body = body
        self.to(ParamConfig.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q


class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body, bias=True):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms, bias=bias))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(ParamConfig.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob


class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body, bias=True):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles, bias=bias))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(ParamConfig.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles


class QLearningOptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(QLearningOptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(ParamConfig.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = torch.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        pi = F.softmax(pi, dim=-1)
        return {'q': q,
                'beta': beta,
                'log_pi': log_pi,
                'pi': pi}
    

class StochasticOptionCriticNet(nn.Module, BaseNet):
    def __init__(
            self, state_dim, action_dim, num_options,
            phi_body=None, pi_body=None, beta_body=None, mu_body=None, q_body=None
    ):
        super(StochasticOptionCriticNet, self).__init__()
        if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
        if mu_body is None: mu_body = FCBody(phi_body.feature_dim, hidden_units=())
        if pi_body is None: pi_body = FCBody(phi_body.feature_dim, hidden_units=())
        if beta_body is None: beta_body = FCBody(phi_body.feature_dim, hidden_units=())
        if q_body is None: q_body = FCBody(phi_body.feature_dim, hidden_units=())
        
        self.phi_body = phi_body
        self.mu_body = mu_body
        self.pi_body = pi_body
        self.beta_body = beta_body
        self.q_body = q_body
        
        self.fc_q = layer_init(nn.Linear(q_body.feature_dim, num_options), 1e-3)
        self.fc_mu = layer_init(nn.Linear(mu_body.feature_dim, num_options), 1e-3)
        self.fc_pi = layer_init(nn.Linear(pi_body.feature_dim, num_options * action_dim), 1e-3)
        self.fc_beta = layer_init(nn.Linear(beta_body.feature_dim, num_options), 1e-3)

        self.mu_params = list(self.mu_body.parameters()) + list(self.fc_mu.parameters()) + list(self.phi_body.parameters())
        self.q_params = list(self.q_body.parameters()) + list(self.fc_q.parameters()) + list(self.phi_body.parameters())
        self.pi_params = list(self.pi_body.parameters()) + list(self.fc_pi.parameters()) + list(self.phi_body.parameters())
        self.beta_params = list(self.beta_body.parameters()) + list(self.fc_beta.parameters()) + list(self.phi_body.parameters())
        
        self.state_dim = state_dim
        self.num_options = num_options
        self.action_dim = action_dim
        self.to(ParamConfig.DEVICE)

    def forward(self, x, flag):
        if flag == 'mu':
            state = tensor(x)
            phi = self.phi_body(state)
            mu_body = self.mu_body(phi)
            # mu_cost = mu_body.abs().sum().unsqueeze(-1).unsqueeze(-1)
            mu_cost = 0
            for param in self.mu_params:
                param_loss = torch.sum(torch.abs(param))
                mu_cost += param_loss
            mu_cost = mu_cost.unsqueeze(-1).unsqueeze(-1)
            logits = self.fc_mu(mu_body)
            probs = 0.9 * F.softmax(logits, dim=1) + 0.1 * torch.ones_like(logits) / logits.shape[1]
            mu_dist = torch.distributions.Categorical(probs=probs)
            # mu_dist = torch.distributions.Categorical(logits=logits)
            return mu_dist, mu_cost
        elif flag == 'beta':
            phi = self.phi_body(x)
            beta_body = self.beta_body(phi)
            # beta_cost = beta_body.abs().sum().unsqueeze(-1).unsqueeze(-1)
            beta_cost = 0
            for param in self.beta_params:
                param_loss = torch.sum(torch.abs(param))
                beta_cost += param_loss
            beta_cost = beta_cost.unsqueeze(-1).unsqueeze(-1)
            logits = self.fc_beta(beta_body)
            probs = 0.8 * torch.sigmoid(logits) + 0.1 * torch.ones_like(logits)
            beta_dist = torch.distributions.bernoulli.Bernoulli(probs=probs)
            # beta_dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
            return beta_dist, beta_cost
        elif flag == 'pi':
            phi = self.phi_body(x)
            pi_body = self.pi_body(phi)
            # pi_cost = pi_body.abs().sum().unsqueeze(-1).unsqueeze(-1)
            pi_cost = 0
            for param in self.pi_params:
                param_loss = torch.sum(torch.abs(param))
                pi_cost += param_loss
            pi_cost = pi_cost.unsqueeze(-1).unsqueeze(-1)
            logits = self.fc_pi(pi_body)
            logits = logits.view(-1, self.num_options, self.action_dim)
            pi_dist = torch.distributions.Categorical(logits=logits)
            return pi_dist, pi_cost
        elif flag == 'q':
            phi = self.phi_body(x)
            q = self.fc_q(self.q_body(phi))
            return q
        else:
            raise NotImplementedError


# class StochasticOptionCriticNetNoShare(nn.Module, BaseNet):
#     def __init__(
#             self, state_dim, action_dim, num_options,
#             phi_body=None, pi_bodies=None, beta_bodies=None, mu_body=None, q_bodies=None
#     ):
#         super(StochasticOptionCriticNetNoShare, self).__init__()
#         if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
#         if mu_body is None: mu_body = FCBody(phi_body.feature_dim, hidden_units=())
#         if pi_bodies is None: pi_bodies = FCBody(phi_body.feature_dim, hidden_units=())
#         if beta_bodies is None: beta_bodies = FCBody(phi_body.feature_dim, hidden_units=())
#         if q_bodies is None: q_bodies = FCBody(phi_body.feature_dim, hidden_units=())
#
#         self.phi_body = phi_body
#         self.mu_body = mu_body
#         self.pi_bodies = pi_bodies
#         self.beta_bodies = beta_bodies
#         self.q_bodies = q_bodies
#
#         # self.fc_q = layer_init(nn.Linear(q_body.feature_dim, num_options), 1e-3)
#         self.fc_qs = nn.ModuleList([layer_init(nn.Linear(q_bodies[i].feature_dim, 1)) for i in range(num_options)])
#         self.fc_mu = layer_init(nn.Linear(mu_body.feature_dim, num_options))
#         self.fc_pis = nn.ModuleList([layer_init(nn.Linear(pi_bodies[i].feature_dim, action_dim)) for i in range(num_options)])
#         self.fc_betas = nn.ModuleList([layer_init(nn.Linear(beta_bodies[i].feature_dim, 1)) for i in range(num_options)])
#
#         self.mu_params = list(self.mu_body.parameters()) + list(self.fc_mu.parameters()) + list(self.phi_body.parameters())
#
#         # self.q_params = [
#         #     list(self.q_bodies[option].parameters())
#         #     + list(self.fc_qs[option].parameters())
#         #     + list(self.phi_body.parameters()) for option in range(num_options)
#         # ]
#
#         self.beta_params = [
#             list(self.beta_bodies[option].parameters())
#             + list(self.fc_betas[option].parameters())
#             + list(self.phi_body.parameters()) for option in range(num_options)
#         ]
#
#         self.pi_params = [
#             list(self.pi_bodies[option].parameters())
#             + list(self.fc_pis[option].parameters())
#             + list(self.phi_body.parameters()) for option in range(num_options)
#         ]
#
#         self.state_dim = state_dim
#         self.num_options = num_options
#         self.action_dim = action_dim
#         self.to(ParamConfig.DEVICE)
#
#     def forward(self, x, flag):
#         phi = self.phi_body(x)
#         if flag == 'mu':
#             mu_body = self.mu_body(phi)
#             # mu_cost = mu_body.pow(2).sum().unsqueeze(-1).unsqueeze(-1)
#             # mu_cost = mu_body.abs().sum().unsqueeze(-1).unsqueeze(-1)
#             mu_cost = 0
#             for param in self.mu_params:
#                 param_loss = torch.sum(torch.abs(param))
#                 mu_cost += param_loss
#             mu_cost = mu_cost.unsqueeze(-1).unsqueeze(-1)
#             logits = self.fc_mu(mu_body)
#             probs = 0.9 * F.softmax(logits, dim=1) + 0.1 * torch.ones_like(logits) / logits.shape[1]
#             mu_dist = torch.distributions.Categorical(probs=probs)
#             # mu_dist = torch.distributions.Categorical(logits=logits)
#             return mu_dist, mu_cost
#         elif flag == 'pi':
#             logits_list = []
#             cost_list = []
#             for option in range(self.num_options):
#                 pi_body = self.pi_bodies[option](phi)
#                 # pi_cost = pi_body.abs().sum().unsqueeze(-1).unsqueeze(-1)
#                 # pi_cost = pi_body.pow(2).sum().unsqueeze(-1).unsqueeze(-1)
#                 cost = 0
#                 for param in self.pi_params[option]:
#                     param_loss = torch.sum(torch.abs(param))
#                     cost += param_loss
#                 cost = cost.unsqueeze(-1).unsqueeze(-1).expand(phi.shape[0], 1)
#                 cost_list.append(cost)
#                 logits = self.fc_pis[option](pi_body).unsqueeze(1)
#                 logits_list.append(logits)
#             cost = torch.cat(cost_list, dim=1)
#             pi_dist = torch.distributions.Categorical(logits=torch.cat(logits_list, dim=1))
#             return pi_dist, cost
#         elif flag == 'beta':
#             logits_list = []
#             cost_list = []
#             for option in range(self.num_options):
#                 beta_body = self.beta_bodies[option](phi)
#                 # beta_cost = beta_body.abs().sum().unsqueeze(-1).unsqueeze(-1)
#                 # beta_cost = beta_body.pow(2).sum().unsqueeze(-1).unsqueeze(-1)
#                 cost = 0
#                 for param in self.beta_params[option]:
#                     param_loss = torch.sum(torch.abs(param))
#                     cost += param_loss
#                 cost = cost.unsqueeze(-1).unsqueeze(-1).expand(phi.shape[0], 1)
#                 cost_list.append(cost)
#                 logits = self.fc_betas[option](beta_body)
#                 logits_list.append(logits)
#             cost = torch.cat(cost_list, dim=1)
#             logits_list = torch.cat(logits_list, dim=1)
#             probs = 0.8 * torch.sigmoid(logits_list) + 0.1 * torch.ones_like(logits_list)
#             beta_dist = torch.distributions.bernoulli.Bernoulli(probs=probs)
#             return beta_dist, cost
#         elif flag == 'q':
#             q_bodies = [self.q_bodies[option](phi) for option in range(self.num_options)]
#             q = torch.cat(([self.fc_qs[option](q_bodies[option]) for option in range(self.num_options)]), dim=1)
#             return q
#         else:
#             raise NotImplementedError


class StochasticOptionCriticNetNoShare(nn.Module, BaseNet):
    def __init__(
            self, state_dim, action_dim, num_options,
            phi_body=None, pi_bodies=None, beta_bodies=None, mu_body=None, q_bodies=None
    ):
        super(StochasticOptionCriticNetNoShare, self).__init__()
        if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
        if mu_body is None: mu_body = FCBody(phi_body.feature_dim, hidden_units=())
        if pi_bodies is None: pi_bodies = FCBody(phi_body.feature_dim, hidden_units=())
        if beta_bodies is None: beta_bodies = FCBody(phi_body.feature_dim, hidden_units=())
        if q_bodies is None: q_bodies = FCBody(phi_body.feature_dim, hidden_units=())
        
        self.phi_body = phi_body
        self.mu_body = mu_body
        self.pi_bodies = pi_bodies
        self.beta_bodies = beta_bodies
        self.q_bodies = q_bodies
        
        self.fc_qs = nn.ModuleList([layer_init(nn.Linear(q_bodies[i].feature_dim, 1)) for i in range(num_options)])
        
        self.fc_mu = layer_init(nn.Linear(mu_body.feature_dim, num_options))
        self.fc_pis = nn.ModuleList(
            [
                layer_init(nn.Linear(pi_bodies[i].feature_dim, action_dim)) for i in range(num_options)
            ]
        )
        self.fc_betas = nn.ModuleList(
            [
                layer_init(nn.Linear(beta_bodies[i].feature_dim, 2)) for i in range(num_options)
            ]
        )
        
        self.mu_params = list(self.mu_body.parameters()) + list(self.fc_mu.parameters()) + list(
            self.phi_body.parameters())
        
        self.q_params = [
            list(self.q_bodies[option].parameters())
            + list(self.fc_qs[option].parameters())
            + list(self.phi_body.parameters()) for option in range(num_options)
        ]
        
        self.beta_params = [
            list(self.beta_bodies[option].parameters())
            + list(self.fc_betas[option].parameters())
            + list(self.phi_body.parameters()) for option in range(num_options)
        ]
        
        self.pi_params = [
            list(self.pi_bodies[option].parameters())
            + list(self.fc_pis[option].parameters())
            + list(self.phi_body.parameters()) for option in range(num_options)
        ]
        
        self.option_parameters = list(itertools.chain.from_iterable(self.pi_params)) + list(
            itertools.chain.from_iterable(self.beta_params))
        self.policy_parameters = self.mu_params
        self.critic_parameters = list(itertools.chain.from_iterable(self.q_params))
        self.sc = to_np(torch.sum(tensor(
            [torch.prod(tensor(p.shape)) for p in self.option_parameters] +
            [torch.prod(tensor(p.shape)) for p in self.policy_parameters]
        ))).item()
        self.state_dim = state_dim
        self.num_options = num_options
        self.action_dim = action_dim
        
        self.pi_beta_fcs = {'pi': self.fc_pis, 'beta': self.fc_betas}
        self.pi_beta_bodies = {'pi': self.pi_bodies, 'beta': self.beta_bodies}
        
        self.to(ParamConfig.DEVICE)
    
    def forward(self, x, flag):
        if flag == 'pi' or flag == 'beta':
            if type(x) == tuple:
                state, option = x
                phi, phi_tc = self.phi_body(state)
                body, body_tc = self.pi_beta_bodies[flag][option](phi)
                logits = self.pi_beta_fcs[flag][option](body)
                dist = torch.distributions.Categorical(logits=logits)
                if flag == 'pi':
                    fc_tc = 2 * self.pi_bodies[option].feature_dim * self.action_dim + self.action_dim
                else:
                    fc_tc = 2 * self.pi_bodies[option].feature_dim * 2 + 2
                tc = phi_tc + body_tc + fc_tc
            else:
                state = x
                logits_list = []
                phi, phi_tc = self.phi_body(state)
                tc = phi_tc
                for option in range(self.num_options):
                    body, body_tc = self.pi_beta_bodies[flag][option](phi)
                    logits = self.pi_beta_fcs[flag][option](body)
                    if flag == 'pi':
                        fc_tc = 2 * self.pi_bodies[option].feature_dim * self.action_dim + self.action_dim
                    else:
                        fc_tc = 2 * self.pi_bodies[option].feature_dim * 2 + 2
                    logits_list.append(logits.unsqueeze(1))
                    tc += body_tc + fc_tc
                logits_matrix = torch.cat(logits_list, dim=1)
                dist = torch.distributions.Categorical(logits=logits_matrix)
                
            return dist, tc
        elif flag == 'q':
            state = x
            phi,_ = self.phi_body(state)
            q_bodies = [self.q_bodies[option](phi)[0] for option in range(self.num_options)]
            q = torch.cat(([self.fc_qs[option](q_bodies[option]) for option in range(self.num_options)]), dim=1)
            return q
        elif flag == 'mu':
            state = x
            phi, phi_tc = self.phi_body(state)
            body, body_tc = self.mu_body(phi)
            logits = self.fc_mu(body)
            dist = torch.distributions.Categorical(logits=logits)
            fc_tc = 2 * self.mu_body.feature_dim * self.num_options + self.num_options
            tc = phi_tc + body_tc + fc_tc
            return dist, tc
        else:
            raise NotImplementedError
    

# class OffOptionMCCOCNet(nn.Module, BaseNet):
#     def __init__(
#             self, state_dim, action_dim, num_options,
#             phi_body=None, pi_bodies=None, beta_bodies=None, mu_body=None, q_bodies=None, return_cost=None
#     ):
#         super(OffOptionMCCOCNet, self).__init__()
#         if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
#         if mu_body is None: mu_body = FCBody(phi_body.feature_dim, hidden_units=())
#         if pi_bodies is None: pi_bodies = FCBody(phi_body.feature_dim, hidden_units=())
#         if beta_bodies is None: beta_bodies = FCBody(phi_body.feature_dim, hidden_units=())
#         if q_bodies is None: q_bodies = FCBody(phi_body.feature_dim, hidden_units=())
#
#         self.phi_body = phi_body
#         self.mu_body = mu_body
#         self.pi_bodies = pi_bodies
#         self.beta_bodies = beta_bodies
#         self.q_bodies = q_bodies
#         self.return_cost = return_cost
#
#         # self.fc_q = layer_init(nn.Linear(q_body.feature_dim, num_options), 1e-3)
#         self.fc_qs = nn.ModuleList([layer_init(nn.Linear(q_bodies[i].feature_dim, 1)) for i in range(num_options)])
#         self.fc_mu = layer_init(nn.Linear(mu_body.feature_dim, num_options))
#         self.fc_pis = nn.ModuleList(
#             [layer_init(nn.Linear(pi_bodies[i].feature_dim, action_dim)) for i in range(num_options)])
#         self.fc_betas = nn.ModuleList(
#             [layer_init(nn.Linear(beta_bodies[i].feature_dim, 1)) for i in range(num_options)])
#
#         self.mu_params = list(self.mu_body.parameters()) + list(self.fc_mu.parameters()) + list(
#             self.phi_body.parameters())
#
#         self.q_params = [
#             list(self.q_bodies[option].parameters())
#             + list(self.fc_qs[option].parameters())
#             + list(self.phi_body.parameters()) for option in range(num_options)
#         ]
#
#         self.beta_params = [
#             list(self.beta_bodies[option].parameters())
#             + list(self.fc_betas[option].parameters())
#             + list(self.phi_body.parameters()) for option in range(num_options)
#         ]
#
#         self.pi_params = [
#             list(self.pi_bodies[option].parameters())
#             + list(self.fc_pis[option].parameters())
#             + list(self.phi_body.parameters()) for option in range(num_options)
#         ]
#
#         self.state_dim = state_dim
#         self.num_options = num_options
#         self.action_dim = action_dim
#
#         self.prob_o = torch.ones((1, num_options)) / num_options
#         self.prob_a = [torch.ones((1, action_dim)) / action_dim for o in range(num_options)]
#         self.prob_o_term = [torch.ones((1, 2)) / 2 for o in range(num_options)]
#
#         self.to(ParamConfig.DEVICE)
#
#     def forward(self, x, flag):
#         if flag == 'mu':
#             state = x
#             phi, phi_cost = self.phi_body(state)
#             mu_body, mu_body_cost = self.mu_body(phi)
#             # tmp = mu_body.unsqueeze(1) * self.fc_mu.weight.unsqueeze(0)
#             # logits = tmp.sum(2) + self.fc_mu.bias
#             logits = self.fc_mu(mu_body)
#             # probs = 0.9 * F.softmax(logits, dim=1) + 0.1 * torch.ones_like(logits) / logits.shape[1]
#             # mu_dist = torch.distributions.Categorical(probs=probs)
#             mu_dist = torch.distributions.Categorical(logits=logits)
#             # if self.return_cost == 'l1_weight':
#             #     mu_fc_cost = self.fc_mu.weight.abs().sum().expand(mu_body.shape[0], 1) + self.fc_mu.bias.abs().sum()
#             # elif self.return_cost == 'l2_weight':
#             #     mu_fc_cost = self.fc_mu.weight.pow(2).sum().expand(mu_body.shape[0], 1) + self.fc_mu.bias.pow(2).sum()
#             #     # neg_mutual_info = torch.mm(mu_dist.probs, torch.log(self.prob_o).t()) + mu_dist.entropy().unsqueeze(-1)
#             #     # mu_fc_cost = mu_fc_cost + 0 * neg_mutual_info
#             #     self.prob_o = 0.99 * self.prob_o + 0.01 * mu_dist.probs.mean(0).unsqueeze(0).data
#             # else:
#             #     raise NotImplementedError
#             # mu_cost = phi_cost + mu_body_cost + mu_fc_cost
#             # mu_cost = mu_cost * self.config.cost_weight
#             mu_cost = torch.ones((x.shape[0], 1)) * 0.0
#             return mu_dist, mu_cost
#         elif flag == 'pi':
#             state = x
#             logits_list = []
#             phi, phi_cost = self.phi_body(state)
#             for option in range(self.num_options):
#                 pi_body, pi_body_cost = self.pi_bodies[option](phi)
#                 # tmp = pi_body.unsqueeze(1) * self.fc_pis[option].weight.unsqueeze(0)
#                 # logits = tmp.sum(2) + self.fc_pis[option].bias
#                 logits = self.fc_pis[option](pi_body)
#                 logits_list.append(logits.unsqueeze(1))
#             logits_matrix = torch.cat(logits_list, dim=1)
#             pi_dist = torch.distributions.Categorical(logits=logits_matrix)
#             # if self.return_cost == 'l1_weight':
#             #     pi_fc_cost = self.fc_pis[option].weight.abs().sum().expand(pi_body.shape[0], 1) + self.fc_pis[option].bias.abs().sum()
#             # elif self.return_cost == 'l2_weight':
#             #     pi_fc_cost = self.fc_pis[option].weight.pow(2).sum().expand(pi_body.shape[0], 1) + self.fc_pis[option].bias.pow(2).sum()
#             # else:
#             #     raise NotImplementedError
#             # pi_cost = phi_cost + pi_body_cost + pi_fc_cost
#             # pi_cost = pi_cost * self.config.cost_weight
#             pi_cost = torch.zeros((x.shape[0], self.num_options))
#             return pi_dist, pi_cost
#         elif flag == 'beta':
#             state = x
#             probs_list = []
#             logits_list = []
#             phi, phi_cost = self.phi_body(state)
#             for option in range(self.num_options):
#                 beta_body, beta_body_cost = self.beta_bodies[option](phi)
#                 # tmp = beta_body.unsqueeze(1) * self.fc_betas[option].weight.unsqueeze(0)
#                 # logits = tmp.sum(2) + self.fc_betas[option].bias
#                 logits = self.fc_betas[option](beta_body)
#                 # probs = 0.8 * torch.sigmoid(logits) + 0.1 * torch.ones_like(logits)
#                 # probs_list.append(probs)
#                 logits_list.append(logits)
#             logits_matrix = torch.cat((logits_list), dim=1)
#             beta_dist = torch.distributions.bernoulli.Bernoulli(logits=logits_matrix)
#             # beta_dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
#             # if self.return_cost == 'l1_weight':
#             #     beta_fc_cost = self.fc_betas[option].weight.abs().sum().expand(beta_body.shape[0], 1) + self.fc_betas[option].bias.abs().sum()
#             # elif self.return_cost == 'l2_weight':
#             #     beta_fc_cost = self.fc_betas[option].weight.pow(2).sum().expand(beta_body.shape[0], 1) + self.fc_betas[option].bias.pow(2).sum()
#             # else:
#             #     raise NotImplementedError
#             # beta_cost = phi_cost + beta_body_cost + beta_fc_cost
#             # beta_cost = beta_cost * self.config.cost_weight
#             beta_cost = torch.zeros((x.shape[0], self.num_options))
#             return beta_dist, beta_cost
#         elif flag == 'q':
#             state = x
#             phi, phi_cost = self.phi_body(state)
#             q_bodies = [self.q_bodies[option](phi) for option in range(self.num_options)]
#             q = torch.cat(([self.fc_qs[option](q_bodies[option]) for option in range(self.num_options)]), dim=1)
#             return q
#         else:
#             raise NotImplementedError


class ValueComplexityOptionCriticNet(nn.Module, BaseNet):
    def __init__(
            self, state_dim, action_dim, num_options,
            cost_weight, mu_entropy_weight, pi_entropy_weight, beta_entropy_weight,
            phi_body=None, pi_bodies=None, beta_bodies=None, mu_body=None, q_bodies=None, return_cost=None,
            lamba=1.0, weight_decay=1.0
    ):
        super(ValueComplexityOptionCriticNet, self).__init__()
        if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
        if mu_body is None: mu_body = FCBody(phi_body.feature_dim, hidden_units=())
        if pi_bodies is None: pi_bodies = FCBody(phi_body.feature_dim, hidden_units=())
        if beta_bodies is None: beta_bodies = FCBody(phi_body.feature_dim, hidden_units=())
        if q_bodies is None: q_bodies = FCBody(phi_body.feature_dim, hidden_units=())
        
        self.cost_weight = cost_weight
        self.mu_entropy_weight = mu_entropy_weight
        self.pi_entropy_weight = pi_entropy_weight
        self.beta_entropy_weight = beta_entropy_weight
        
        self.phi_body = phi_body
        self.mu_body = mu_body
        self.pi_bodies = pi_bodies
        self.beta_bodies = beta_bodies
        self.q_bodies = q_bodies
        self.return_cost = return_cost
        
        self.fc_qs = nn.ModuleList([layer_init(nn.Linear(q_bodies[i].feature_dim, 1)) for i in range(num_options)])
        if self.return_cost == 'l0':
            droprate_init = 0.5
            temperature = 2./3.
            local_rep = False
            self.N = 50000
            self.weight_decay = self.N * weight_decay
            self.fc_mu = L0Dense(
                mu_body.feature_dim, num_options,
                droprate_init=droprate_init, weight_decay=self.weight_decay,
                lamba=lamba, local_rep=local_rep, temperature=temperature
            )
            self.fc_pis = nn.ModuleList(
                [
                    L0Dense(
                        pi_bodies[i].feature_dim, action_dim,
                        droprate_init=droprate_init, weight_decay=self.weight_decay,
                        lamba=lamba, local_rep=local_rep, temperature=temperature) for i in range(num_options)
                ]
            )
            self.fc_betas = nn.ModuleList(
                [
                    L0Dense(
                        beta_bodies[i].feature_dim, 2,
                        droprate_init=droprate_init, weight_decay=self.weight_decay,
                        lamba=lamba, local_rep=local_rep, temperature=temperature) for i in range(num_options)
                ]
            )
        else:
            self.fc_mu = layer_init(nn.Linear(mu_body.feature_dim, num_options))
            self.fc_pis = nn.ModuleList(
                [
                    layer_init(nn.Linear(pi_bodies[i].feature_dim, action_dim)) for i in range(num_options)
                ]
            )
            self.fc_betas = nn.ModuleList(
                [
                    layer_init(nn.Linear(beta_bodies[i].feature_dim, 2)) for i in range(num_options)
                ]
            )
        
        self.mu_params = list(self.mu_body.parameters()) + list(self.fc_mu.parameters()) + list(
            self.phi_body.parameters())
        
        self.q_params = [
            list(self.q_bodies[option].parameters())
            + list(self.fc_qs[option].parameters())
            + list(self.phi_body.parameters()) for option in range(num_options)
        ]
        
        self.beta_params = [
            list(self.beta_bodies[option].parameters())
            + list(self.fc_betas[option].parameters())
            + list(self.phi_body.parameters()) for option in range(num_options)
        ]
        
        self.pi_params = [
            list(self.pi_bodies[option].parameters())
            + list(self.fc_pis[option].parameters())
            + list(self.phi_body.parameters()) for option in range(num_options)
        ]

        self.option_parameters = list(itertools.chain.from_iterable(self.pi_params)) + list(
            itertools.chain.from_iterable(self.beta_params))
        self.policy_parameters = self.mu_params
        self.critic_parameters = list(itertools.chain.from_iterable(self.q_params))
        
        self.state_dim = state_dim
        self.num_options = num_options
        self.action_dim = action_dim

        self.pi_beta_fcs = {'pi': self.fc_pis, 'beta': self.fc_betas}
        self.pi_beta_bodies = {'pi': self.pi_bodies, 'beta': self.beta_bodies}
        
        self.to(ParamConfig.DEVICE)
        
    def reinitialize_option(self, option):
        # for layer in self.q_bodies[option].layers:
        #     layer_init(layer)
        # layer_init(self.fc_qs[option])
        for layer in self.pi_bodies[option].layers:
            layer_init(layer)
        layer_init(self.fc_pis[option])
        for layer in self.beta_bodies[option].layers:
            layer_init(layer)
        layer_init(self.fc_betas[option])
        for layer in self.mu_body.layers:
            layer_init(layer)
        layer_init(self.fc_mu)
        
    def forward(self, x, flag):
        if flag == 'pi' or flag == 'beta':
            if type(x) == tuple:
                state, option = x
                phi, phi_cost = self.phi_body(state)
                probs, cost = self.forward_helper(phi, option, flag)
                dist = torch.distributions.Categorical(probs=probs)
                entropy = dist.entropy().unsqueeze(-1)
            else:
                state = x
                probs_list = []
                cost_list = []
                phi, phi_cost = self.phi_body(state)
                for option in range(self.num_options):
                    probs, cost = self.forward_helper(phi, option, flag)
                    probs_list.append(probs.unsqueeze(1))
                    cost_list.append(cost)
                probs_matrix = torch.cat(probs_list, dim=1)
                dist = torch.distributions.Categorical(probs=probs_matrix)
                cost = torch.cat(cost_list, dim=1)
                entropy = dist.entropy()
            cost = phi_cost + cost
            return dist, cost, entropy
        elif flag == 'q':
            state = x
            phi, phi_cost = self.phi_body(state)
            q_bodies = [self.q_bodies[option](phi) for option in range(self.num_options)]
            q = torch.cat(([self.fc_qs[option](q_bodies[option]) for option in range(self.num_options)]), dim=1)
            return q
        elif flag == 'mu':
            state = x
            phi, phi_cost = self.phi_body(state)
            body, body_cost = self.mu_body(phi)
            if self.return_cost == 'l1_multiplication':
                input_weight_prod = body.unsqueeze(1) * self.fc_mu.weight.unsqueeze(0)
                logits = input_weight_prod.sum(2) + self.fc_mu.bias
                fc_cost = input_weight_prod.abs().sum(dim=1).sum(dim=1).unsqueeze(-1) + self.fc_mu.bias.abs().sum()
            else:
                logits = self.fc_mu(body)
                fc_cost = self.get_fc_cost(self.fc_mu, logits.shape[0])
            probs = torch.softmax(logits, dim=1)
            # probs = 0.9 * torch.softmax(logits, dim=1) + 0.1 * torch.ones_like(logits) / logits.shape[1]
            dist = torch.distributions.Categorical(probs=probs)
            cost = phi_cost + body_cost + fc_cost
            entropy = dist.entropy().unsqueeze(-1)
            return dist, cost, entropy
        else:
            raise NotImplementedError

    def forward_helper(self, phi, o, flag):
        body, body_cost = self.pi_beta_bodies[flag][o](phi)
        if self.return_cost == 'l1_multiplication':
            input_weight_prod = body.unsqueeze(1) * self.pi_beta_fcs[flag][o].weight.unsqueeze(0)
            logits = input_weight_prod.sum(2) + self.pi_beta_fcs[flag][o].bias
            fc_cost = input_weight_prod.abs().sum(dim=1).sum(dim=1).unsqueeze(-1) + self.pi_beta_fcs[flag][o].bias.abs().sum()
        else:
            logits = self.pi_beta_fcs[flag][o](body)
            fc_cost = self.get_fc_cost(self.pi_beta_fcs[flag][o], logits.shape[0])
        # probs = 0.9 * torch.softmax(logits, dim=1) + 0.1 * torch.ones_like(logits) / logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        return probs, body_cost + fc_cost
    
    def get_fc_cost(self, fc, input_dim):
        if self.return_cost == 'l1_neuron':
            fc_cost = 0
        elif self.return_cost == 'l2_neuron':
            fc_cost = 0
        elif self.return_cost == 'sparse_neuron':
            fc_cost = 0
        elif self.return_cost == 'l1_weight':
            fc_cost = fc.weight.abs().sum().expand(input_dim, 1) + fc.bias.abs().sum()
        elif self.return_cost == 'l2_weight':
            fc_cost = fc.weight.pow(2).sum().expand(input_dim, 1) + fc.bias.pow(2).sum()
        elif self.return_cost == 'l0':
            fc_cost = - (1. / self.N) * fc.regularization().expand(input_dim, 1)
        elif self.return_cost == 'non_zero_weights':
            fc_cost = (fc.weight.abs() > 0.001).sum().type(torch.FloatTensor).expand(input_dim, 1)
        elif self.return_cost == 'non_zero_neurons':
            fc_cost = 0
        else:
            raise NotImplementedError
        return fc_cost
        
        
# class DeterministicOptionCriticNet(nn.Module, BaseNet):
#     def __init__(
#             self, state_dim, action_dim, option_feature_dim,
#             phi_body=None, pi_body=None, beta_body=None, mu_body=None, q_body=None, mine_body=None
#     ):
#         super(DeterministicOptionCriticNet, self).__init__()
#         if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
#         if mu_body is None: mu_body = FCBody(phi_body.feature_dim, hidden_units=())
#         if pi_body is None: pi_body = FCBodyWithTwoInputs(phi_body.feature_dim, option_feature_dim, hidden_units=())
#         if beta_body is None: beta_body = FCBodyWithTwoInputs(state_dim, option_feature_dim, hidden_units=())
#         if q_body is None: q_body = FCBodyWithTwoInputs(state_dim, option_feature_dim, hidden_units=())
#
#         self.option_feature_dim = option_feature_dim
#
#         self.phi_body = phi_body
#         self.pi_body = pi_body
#         self.beta_body = beta_body
#         self.mu_body = mu_body
#         self.q_body = q_body
#         self.mine_body = mine_body
#
#         self.option_feature_dim_divide_2 = int(option_feature_dim / 2)
#         self.fc_q = layer_init(nn.Linear(q_body.feature_dim, self.option_feature_dim), 1e-3)
#         self.fc_mu = layer_init(nn.Linear(mu_body.feature_dim, option_feature_dim), 1e-3)
#         self.fc_pi = layer_init(nn.Linear(pi_body.feature_dim, action_dim * self.option_feature_dim_divide_2), 1e-3)
#         self.fc_beta = layer_init(nn.Linear(beta_body.feature_dim, self.option_feature_dim_divide_2), 1e-3)
#         self.fc_mine = layer_init(nn.Linear(mine_body.feature_dim, 1), 1e-3)
#
#         self.action_dim = action_dim
#
#         self.mu_params = list(self.mu_body.parameters()) + list(self.fc_mu.parameters()) + list(self.phi_body.parameters())
#         self.q_params = list(self.q_body.parameters()) + list(self.fc_q.parameters()) + list(self.phi_body.parameters())
#         self.pi_parameters = list(self.pi_body.parameters()) + list(self.fc_pi.parameters()) + list(self.phi_body.parameters())
#         self.beta_parameters = list(self.beta_body.parameters()) + list(self.fc_beta.parameters()) + list(self.phi_body.parameters())
#         self.mine_parameters = list(self.mine_body.parameters()) + list(self.fc_mine.parameters())
#
#         self.to(ParamConfig.DEVICE)
#
#     def forward(self, x, flag):
#         if flag == 'mu':
#             phi = self.phi_body(tensor(x))
#             mu_body_output = self.mu_body(phi)
#             logits = self.fc_mu(mu_body_output)
#             dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
#             option = dist.probs
#             entropy = dist.entropy()
#             return logits, entropy
#         elif flag == 'q':
#             # state, option = x
#             # phi = self.phi_body(tensor(state))
#             # q = torch.bmm(
#             #     self.fc_q(self.q_body(phi)).view(-1, 1, self.option_feature_dim),
#             #     option.view(-1, self.option_feature_dim, 1)
#             # )
#             # return q
#             state, option = x
#             phi = self.phi_body(tensor(state))
#             q = self.fc_q(self.q_body(phi, tensor(option)))
#             return q
#         elif flag == 'pi':
#             state, option = x
#             phi = self.phi_body(tensor(state))
#             logits = torch.bmm(
#                 self.fc_pi(self.pi_body(phi)).view(-1, self.action_dim, self.option_feature_dim_divide_2),
#                 option[:, :self.option_feature_dim_divide_2].view(-1, self.option_feature_dim_divide_2, 1)
#             ).view(-1, self.action_dim)
#             dist = torch.distributions.Categorical(logits=logits)
#             return dist
#
#         elif flag == 'beta':
#             state, option = x
#             phi = self.phi_body(tensor(state))
#             logits = torch.bmm(
#                 self.fc_beta(self.beta_body(phi)).view(-1, 1, self.option_feature_dim_divide_2),
#                 option[:, self.option_feature_dim_divide_2:].view(-1, self.option_feature_dim_divide_2, 1)
#             ).view(-1, 1)
#             dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
#
#             return dist
#         elif flag == 'mine':
#             input1, input2 = x
#             mine = self.fc_mine(self.mine_body(tensor(input1), input2))
#
#             return mine
#         else:
#             raise NotImplementedError


# class DeterministicOptionCriticNet(nn.Module, BaseNet):
#     def __init__(
#             self, state_dim, action_dim, option_dim,
#             pi_phi_body=None, beta_phi_body=None, q_phi_body=None,
#             mu_body=None, pi_body=None, beta_body=None, q_body=None
#     ):
#         super(DeterministicOptionCriticNet, self).__init__()
#         if pi_phi_body is None: pi_phi_body = FCBody(state_dim, hidden_units=())
#         if beta_phi_body is None: beta_phi_body = FCBody(state_dim, hidden_units=())
#         if q_phi_body is None: q_phi_body = FCBody(state_dim, hidden_units=())
#         if mu_body is None: mu_body = FCBody(state_dim, hidden_units=())
#         if pi_body is None: pi_body = FCBody(mu_body.feature_dim, hidden_units=())
#         if beta_body is None: beta_body = FCBody(mu_body.feature_dim, hidden_units=())
#         if q_body is None: q_body = FCBody(mu_body.feature_dim, hidden_units=())
#         # if pi_body is None: pi_body = FCBody(pi_phi_body.feature_dim + mu_body.feature_dim, hidden_units=())
#         # if beta_body is None: beta_body = FCBody(beta_phi_body.feature_dim + mu_body.feature_dim, hidden_units=())
#         # if q_body is None: q_body = FCBody(q_phi_body.feature_dim + mu_body.feature_dim, hidden_units=())
#
#         self.pi_phi_body = pi_phi_body
#         self.beta_phi_body = beta_phi_body
#         self.q_phi_body = q_phi_body
#         self.mu_body = mu_body
#         self.pi_body = pi_body
#         self.beta_body = beta_body
#         self.q_body = q_body
#         # self.mine_body = mine_body
#
#         self.fc_q = layer_init(nn.Linear(q_body.feature_dim, 1), 1e-3)
#         self.fc_pi = layer_init(nn.Linear(pi_body.feature_dim, action_dim), 1e-3)
#         self.fc_beta = layer_init(nn.Linear(beta_body.feature_dim, 1), 1e-3)
#         # self.fc_mine = layer_init(nn.Linear(mine_body.feature_dim, option_feature_dim), 1e-3)
#
#         # self.dropout_layer = torch.nn.Dropout(p=0.5)
#
#         self.action_dim = action_dim
#
#         self.q_params = list(self.q_body.parameters()) + list(self.fc_q.parameters()) + list(self.q_phi_body.parameters())
#         self.pi_beta_params = list(self.pi_phi_body.parameters()) + list(self.pi_body.parameters()) + \
#                               list(self.fc_pi.parameters()) + list(self.beta_phi_body.parameters()) + \
#                               list(self.beta_body.parameters()) + list(self.fc_beta.parameters())
#         # self.state_phi_params = list(self.state_phi_body.parameters())
#         self.mu_params = list(self.mu_body.parameters())
#         # self.mine_params = list(self.mine_body.parameters()) + list(self.fc_mine.parameters())
#
#         self.to(ParamConfig.DEVICE)
#
#     def forward(self, x, flag):
#         if flag == 'mu':
#             option_features = self.mu_body(x)
#             # m = torch.distributions.normal.Normal(torch.zeros_like(option_features), torch.ones_like(option_features))
#             # option_features = option_features + m.sample() * 0.5
#             option_features = torch.sigmoid(option_features)
#             return option_features
#         elif flag == 'mu test':
#             option_features = self.mu_body(x)
#             option_features = torch.sigmoid(option_features)
#             return option_features
#         elif flag == 'q':
#             state, option_features = x
#             state_features = self.q_phi_body(state)
#             # q = self.fc_q(self.q_body(torch.cat((state_features, option_features), dim=1)))
#             q = self.fc_q(self.q_body(state_features * option_features))
#             return q
#         elif flag == 'pi':
#             state, option_features = x
#             state_features = self.pi_phi_body(state)
#             # state_features = self.dropout_layer(state_features)
#             # logits = self.fc_pi(self.pi_body(torch.cat((state_features, option_features), dim=1)))
#             logits = self.fc_pi(self.pi_body(state_features * option_features))
#             dist = torch.distributions.Categorical(logits=logits)
#             return dist
#         elif flag == 'beta':
#             state, option_features = x
#             state_features = self.beta_phi_body(state)
#             # logits = self.fc_beta(self.beta_body(torch.cat((state_features, option_features), dim=1)))
#             logits = self.fc_beta(self.beta_body(state_features * option_features))
#             dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
#             return dist
#         # elif flag == 'mine':
#         #     input1, input2 = x
#         #     mine = self.fc_mine(self.mine_body(tensor(input1), input2))
#         #     return mine
#         elif flag == 'pi phi':
#             state_features = self.pi_phi_body(x)
#             return state_features
#         else:
#             raise NotImplementedError


# class GeneralOptionCriticNet(nn.Module, BaseNet):
#     def __init__(
#             self, state_dim, action_dim, pi_option_dim, beta_option_dim,
#             pi_so_body=None, beta_so_body=None, q_so_body=None, mine_so_body=None,
#             mu_body=None, pi_body=None, beta_body=None, q_body=None, mine_body=None
#     ):
#         super(GeneralOptionCriticNet, self).__init__()
#         if mu_body is None: mu_body = FCBodyWithTwoInputs(state_dim, state_dim, combine='concatenate', hidden_units1=(), hidden_units2=())
#         if pi_so_body is None: pi_so_body = FCBodyWithTwoInputs(state_dim, mu_body.n1.feature_dim, hidden_units1=(), hidden_units2=())
#         if beta_so_body is None: beta_so_body = FCBodyWithTwoInputs(state_dim, mu_body.n2.feature_dim, hidden_units1=(), hidden_units2=())
#         if q_so_body is None: q_so_body = FCBodyWithTwoInputs(state_dim, mu_body.feature_dim, hidden_units1=(), hidden_units2=())
#         if mine_so_body is None: mine_so_body = FCBodyWithTwoInputs(mu_body.n1.feature_dim, action_dim, hidden_units1=(), hidden_units2=())
#         if pi_body is None: pi_body = FCBody(pi_so_body.feature_dim, hidden_units=())
#         if beta_body is None: beta_body = FCBody(beta_so_body.feature_dim, hidden_units=())
#         if q_body is None: q_body = FCBody(q_so_body.feature_dim, hidden_units=())
#         if mine_body is None: mine_body = FCBody(mine_so_body.feature_dim, hidden_units=())
#
#         self.action_dim = action_dim
#         self.pi_option_dim = pi_option_dim
#         self.beta_option_dim = beta_option_dim
#
#         self.pi_so_body = pi_so_body
#         self.beta_so_body = beta_so_body
#         self.q_so_body = q_so_body
#         self.mine_so_body = mine_so_body
#         self.mu_body = mu_body
#         self.pi_body = pi_body
#         self.beta_body = beta_body
#         self.q_body = q_body
#         self.mine_body = mine_body
#         self.mu_pi_fc = layer_init(nn.Linear(mu_body.n1.feature_dim, pi_option_dim), 1e-3)
#         self.mu_beta_fc = layer_init(nn.Linear(mu_body.n2.feature_dim, beta_option_dim), 1e-3)
#         self.pi_fc = layer_init(nn.Linear(pi_body.feature_dim, action_dim), 1e-3)
#         self.beta_fc = layer_init(nn.Linear(beta_body.feature_dim, 1), 1e-3)
#         self.q_fc = layer_init(nn.Linear(q_body.feature_dim, 1), 1e-3)
#         self.mine_fc = layer_init(nn.Linear(mine_body.feature_dim, 1), 1e-3)
#
#         self.q_params = list(self.q_so_body.parameters()) + list(self.q_body.parameters()) + list(self.q_fc.parameters())
#         self.pi_beta_params = list(self.pi_so_body.parameters()) + \
#                               list(self.pi_body.parameters()) + \
#                               list(self.beta_so_body.parameters()) + \
#                               list(self.beta_body.parameters()) + list(self.beta_fc.parameters()) + list(self.pi_fc.parameters())
#         self.mu_params = list(self.mu_body.parameters()) + list(self.mu_pi_fc.parameters()) + list(self.mu_beta_fc.parameters())
#         self.mine_params = list(self.mine_so_body.parameters()) + list(self.mine_body.parameters()) + list(self.mine_fc.parameters())
#
#         self.to(ParamConfig.DEVICE)
#
#     def forward(self, x, flag):
#         if flag == 'mu':
#             # dist = torch.distributions.normal.Normal(torch.zeros_like(x), torch.ones_like(x))
#             # y = dist.sample() * 1
#             # y = torch.zeros_like(x)
#             # x = torch.cat((x, y), dim=1)
#             option_features = self.mu_body(x, x)
#             option_features = torch.cat(
#                 (
#                     torch.tanh(self.mu_pi_fc(option_features[:, :self.mu_body.n1.feature_dim])),
#                     torch.tanh(self.mu_beta_fc(option_features[:, self.mu_body.n1.feature_dim:]))
#                 ), dim=1
#             )
#             # dist = torch.distributions.normal.Normal(torch.zeros_like(option_features), torch.ones_like(option_features))
#             # y = dist.sample() * 0.5
#             # option_features += y
#             # option_features = self.mu_pi_fc(option_features[:, :self.mu_body.n1.feature_dim])
#             # option_features = torch.tanh(option_features)
#             return option_features
#         elif flag == 'pi':
#             state, option_features = x
#             pi_option_features = option_features[:, :self.pi_option_dim]
#             logits = self.pi_fc(self.pi_body(self.pi_so_body(state, pi_option_features)))
#             rtv = torch.distributions.Categorical(logits=logits)
#             return rtv
#         # elif flag == 'pi':
#         #     state, option_features = x
#         #     logits = option_features[:, :self.pi_option_dim]
#         #     rtv = torch.distributions.Categorical(logits=logits)
#         #     return rtv
#         elif flag == 'beta':
#             state, option_features = x
#             beta_option_features = option_features[:, self.pi_option_dim:]
#             logits = self.beta_fc(self.beta_body(self.beta_so_body(state, beta_option_features)))
#             # logits = torch.ones_like(logits) * 100
#             rtv = torch.distributions.bernoulli.Bernoulli(logits=logits)
#             return rtv
#         elif flag == 'q':
#             state, option_features = x
#             logits = self.q_fc(self.q_body(self.q_so_body(state, option_features)))
#             return logits
#         elif flag == 'mine':
#             input1, input2 = x
#             input1 = input1[:, :self.pi_option_dim]
#             mine = self.mine_fc(self.mine_body(self.mine_so_body(input1, input2)))
#             return mine
#         else:
#             raise NotImplementedError
        

# class DeterministicOptionCriticNet(nn.Module, BaseNet):
#     def __init__(
#             self, state_dim, action_dim, option_feature_dim,
#             phi_body=None, pi_body=None, beta_body=None, mu_body=None, q_body=None, mine_body=None
#     ):
#         super(DeterministicOptionCriticNet, self).__init__()
#         if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
#         if mu_body is None: mu_body = FCBody(phi_body.feature_dim, hidden_units=())
#         if pi_body is None: pi_body = FCBodyWithTwoInputs(phi_body.feature_dim, option_feature_dim, hidden_units=())
#         if beta_body is None: beta_body = FCBodyWithTwoInputs(state_dim, option_feature_dim, hidden_units=())
#         if q_body is None: q_body = FCBodyWithTwoInputs(state_dim, option_feature_dim, hidden_units=())
#
#         self.option_feature_dim = option_feature_dim
#
#         self.phi_body = phi_body
#         self.pi_body = pi_body
#         self.beta_body = beta_body
#         self.mu_body = mu_body
#         self.q_body = q_body
#         self.mine_body = mine_body
#
#         self.fc_q = layer_init(nn.Linear(q_body.feature_dim, 1), 1e-3)
#         self.fc_mu = layer_init(nn.Linear(mu_body.feature_dim, option_feature_dim), 1e-3)
#         self.fc_pi = layer_init(nn.Linear(pi_body.feature_dim, action_dim), 1e-3)
#         self.fc_beta = layer_init(nn.Linear(beta_body.feature_dim, 1), 1e-3)
#         self.fc_mine = layer_init(nn.Linear(mine_body.feature_dim, 1), 1e-3)
#
#         self.action_dim = action_dim
#
#         self.mu_params = list(self.mu_body.parameters()) + list(self.fc_mu.parameters()) + list(
#             self.phi_body.parameters())
#         self.q_params = list(self.q_body.parameters()) + list(self.fc_q.parameters()) + list(self.phi_body.parameters())
#         self.pi_parameters = list(self.pi_body.parameters()) + list(self.fc_pi.parameters()) + list(
#             self.phi_body.parameters())
#         self.beta_parameters = list(self.beta_body.parameters()) + list(self.fc_beta.parameters()) + list(
#             self.phi_body.parameters())
#         self.mine_parameters = list(self.mine_body.parameters()) + list(self.fc_mine.parameters())
#
#         self.to(ParamConfig.DEVICE)
#
#     def forward(self, x, flag):
#         if flag == 'mu':
#             phi = self.phi_body(tensor(x))
#             mu_body_output = self.mu_body(phi)
#             logits = self.fc_mu(mu_body_output)
#             # option = torch.relu(fc_mu_output)
#             # option = torch.tanh(fc_mu_output)
#             # option = torch.softmax(fc_mu_output, dim=1)
#             # option = torch.sigmoid(fc_mu_output)
#             dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
#             option = dist.probs
#             m = torch.distributions.normal.Normal(torch.tensor([0.0 for _ in range(self.option_feature_dim)]), torch.tensor([0.1 for _ in range(self.option_feature_dim)]))
#             option = option + m.sample()
#             entropy = dist.entropy()
#             return option, entropy
#         elif flag == 'q':
#             state, option = x
#             phi = self.phi_body(tensor(state))
#             q = self.fc_q(self.q_body(phi, tensor(option)))
#             return q
#         elif flag == 'pi':
#             state, option = x
#             phi = self.phi_body(tensor(state))
#             logits = self.fc_pi(self.pi_body(phi, tensor(option)))
#             dist = torch.distributions.Categorical(logits=logits)
#             return dist
#
#         elif flag == 'beta':
#             state, option = x
#             phi = self.phi_body(tensor(state))
#             logits = self.fc_beta(self.beta_body(phi, tensor(option)))
#             dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
#
#             return dist
#         elif flag == 'mine':
#             input1, input2 = x
#             mine = self.fc_mine(self.mine_body(tensor(input1), input2))
#
#             return mine
#         else:
#             raise NotImplementedError
        

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
        if actor_body is None: actor_body = FCBody(phi_body.feature_dim, hidden_units=())
        if critic_body is None: critic_body = FCBody(phi_body.feature_dim, hidden_units=())
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.to(ParamConfig.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.fc_action(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(phi, a))


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
        if actor_body is None: actor_body = FCBody(phi_body.feature_dim, hidden_units=())
        if critic_body is None: critic_body = FCBody(phi_body.feature_dim, hidden_units=())
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(ParamConfig.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        mean = torch.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v}


class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
        if actor_body is None: actor_body = FCBody(phi_body.feature_dim, hidden_units=())
        if critic_body is None: critic_body = FCBody(phi_body.feature_dim, hidden_units=())
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.to(ParamConfig.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        logits = self.fc_action(phi_a)
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        max_a_prob = dist.probs.max(dim=1).values.unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action.unsqueeze(-1),
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v,
                'max_a_prob': max_a_prob}


class CategoricalActorQCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorQCriticNet, self).__init__()
        if phi_body is None: phi_body = FCBody(state_dim, hidden_units=())
        if actor_body is None: actor_body = FCBody(phi_body.feature_dim, hidden_units=())
        if critic_body is None: critic_body = FCBody(phi_body.feature_dim, hidden_units=())
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, action_dim), 1e-3)
        
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.to(ParamConfig.DEVICE)
    
    def forward(self, obs, flag):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        if flag == 'actor':
            phi_a = self.actor_body(phi)
            logits = self.fc_action(phi_a)
            return logits
        elif flag == 'q values':
            phi_q = self.critic_body(phi)
            q_values = self.fc_critic(phi_q)
            return q_values
        elif flag == 'both':
            phi_a = self.actor_body(phi)
            logits = self.fc_action(phi_a)
            phi_q = self.critic_body(phi)
            q_values = self.fc_critic(phi_q)
            return logits, q_values
        else:
            raise NotImplementedError