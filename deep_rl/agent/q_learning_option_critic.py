#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class OptionCritic(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn(config.seed)
        self.network = config.network
        
        if config.use_target_network:
            self.target_network = config.network_fn()
            self.target_network.load_state_dict(self.network.state_dict())
        else:
            self.target_network = self.network
            
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.total_steps = 0

        self.states = self.task.reset()
        self.is_initial_states = tensor(np.ones((1))).byte()
        self.prev_options = self.is_initial_states.clone().long()

    def sample_option(self, prediction, epsilon, prev_option, is_intial_states):
        with torch.no_grad():
            q_option = prediction['q']
            pi_option = torch.zeros_like(q_option).add(epsilon / q_option.size(1))
            greedy_option = q_option.argmax(dim=-1, keepdim=True)
            prob = 1 - epsilon + epsilon / q_option.size(1)
            prob = torch.zeros_like(pi_option).add(prob)
            pi_option.scatter_(1, greedy_option, prob)
        
            mask = torch.zeros_like(q_option)
            mask[:, prev_option] = 1
            beta = prediction['beta']
            pi_hat_option = (1 - beta) * mask + beta * pi_option
        
            dist = torch.distributions.Categorical(probs=pi_option)
            options = dist.sample()
            dist = torch.distributions.Categorical(probs=pi_hat_option)
            options_hat = dist.sample()
        
            options = torch.where(is_intial_states, options, options_hat)
        return options

    def step(self):
        report = {}
        config = self.config
        storage = Storage(config.rollout_length, ['beta', 'o', 'beta_adv', 'prev_o', 'init', 'eps'])

        for _ in range(config.rollout_length):
            # obtain option values, intra-option policies and termination conditions for current state
            prediction = self.network(np.expand_dims(config.state_normalizer(self.states), axis=0))
            
            # choose epsilon for option selection
            epsilon = config.random_option_prob(1)
            
            # stick on old option or choose new option
            options = self.sample_option(prediction, epsilon, self.prev_options, self.is_initial_states)
            
            prediction['pi'] = prediction['pi'][0][options]
            prediction['log_pi'] = prediction['log_pi'][0][options]
            dist = torch.distributions.Categorical(probs=prediction['pi'])
            
            # choose action
            actions = dist.sample()
            entropy = dist.entropy()

            next_states, rewards, terminals, info = self.task.step(to_np(actions).item())
            
            # record info
            ret = info['episodic_return']
            if ret is not None:
                report.setdefault('episodic_return', []).append(ret)
                report.setdefault('episodic_length', []).append(info['episodic_length'])
                
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'o': options.unsqueeze(-1),
                         'prev_o': self.prev_options.unsqueeze(-1),
                         'ent': entropy.unsqueeze(-1),
                         'a': actions.unsqueeze(-1),
                         'init': self.is_initial_states.unsqueeze(-1).float(),
                         'eps': epsilon})

            self.is_initial_states = tensor(terminals).byte().unsqueeze(-1)
            self.prev_options = options
            self.states = next_states
            self.total_steps += 1

            if config.use_target_network and self.total_steps / self.config.sgd_update_frequency % self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

        with torch.no_grad():
            prediction = self.target_network(np.expand_dims(config.state_normalizer(self.states), axis=0))
            storage.placeholder()
            betas = prediction['beta'][0, self.prev_options]
            ret = (1 - betas) * prediction['q'][0, self.prev_options] + betas * torch.max(prediction['q'], dim=-1).values
            ret = ret.unsqueeze(-1)

        for i in reversed(range(config.rollout_length)):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            adv = ret - storage.q[i].gather(1, storage.o[i])
            storage.ret[i] = ret
            storage.adv[i] = adv

            v = storage.q[i].max(dim=-1, keepdim=True)[0] * (1 - storage.eps[i]) + storage.q[i].mean(-1).unsqueeze(-1) * storage.eps[i]
            q = storage.q[i].gather(1, storage.prev_o[i])
            storage.beta_adv[i] = q - v + config.termination_regularizer

        q, beta, log_pi, ret, adv, beta_adv, ent, option, action, initial_states, prev_o = \
            storage.cat(['q', 'beta', 'log_pi', 'ret', 'adv', 'beta_adv', 'ent', 'o', 'a', 'init', 'prev_o'])

        q_loss = (q.gather(1, option) - ret.detach()).pow(2).mul(0.5).mean()
        pi_loss = -(log_pi.gather(1, action) * adv.detach()) - config.entropy_weight * ent
        pi_loss = pi_loss.mean()
        beta_loss = (beta.gather(1, prev_o) * beta_adv.detach() * (1 - initial_states)).mean()

        self.optimizer.zero_grad()
        (pi_loss + q_loss + beta_loss).backward()
        if self.config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        
        return report
