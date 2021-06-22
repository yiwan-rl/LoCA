#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from ..utils.torch_utils import tensor, range_tensor, to_np, set_optimizer
from ..utils.misc import close_obj
from .BaseAgent import BaseActor, BaseAgent


class QLearningActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        
        self.total_achieved_goals = 0

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config

        if hasattr(config, "async_actor") and config.async_actor is True:
            with config.lock:
                q_values = self._network(np.expand_dims(config.state_normalizer(self._state), axis=0))
        else:
            q_values = self._network(np.expand_dims(config.state_normalizer(self._state), axis=0))

        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step(action)
        
        # if info['goal'] is True:
        #     self.total_achieved_goals += 1
        
        if config.bootstrap_from_timeout is True and info['TimeLimit.truncated'] is True:
            done = False
        entry = [self._state, action, reward, next_state, int(done), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class QLearning(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = QLearningActor(config)

        self.network = config.network
        if config.use_target_network:
            self.target_network = config.network_fn()
            self.target_network.load_state_dict(self.network.state_dict())
        else:
            self.target_network = self.network
        self.optimizer = set_optimizer(self.network.parameters(), config)

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(np.expand_dims(state, axis=0))
        action = to_np(q.argmax(-1))[0]
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        # if self.total_steps % 10000 == 0:
        #     print(self.network.fc_head.weight)
        report = {}
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            # self.record_online_return(info)
            ret = info['episodic_return']
            if ret is not None:
                report.setdefault('episodic_return', []).append(ret)
                report.setdefault('episodic_length', []).append(info['episodic_length'])
                report.setdefault('achieved_goals', []).append(self.actor.total_achieved_goals)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            if self.config.residual_gradient or self.config.reg_Q:
                q_next = self.target_network(next_states)
            else:
                q_next = self.target_network(next_states).detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1).values
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            target = q_next + rewards
            actions = tensor(actions).long()
            q = self.network(states)
            q = q[self.batch_indices, actions]
            if self.config.reg_Q:
                loss = (target.detach() - q).pow(2).mul(0.5).mean() + 0.5 * (self.config.discount ** 2) * q_next.pow(2).mul(0.5).mean() - 0.5 * (self.config.discount ** 2) * q.pow(2).mul(0.5).mean()
            else:
                loss = (target - q).pow(2).mul(0.5).mean()

            report.setdefault('q_loss', []).append(to_np(loss))
            report.setdefault('q_values', []).append(to_np(q).mean())
            
            self.optimizer.zero_grad()
            loss.backward()
            # if self.config.residual_gradient:
            #     total_norm = 0
            #     for p in self.network.parameters():
            #         param_norm = (p.grad.data / (target - q).data).norm(2)
            #         total_norm += param_norm.item() ** 2
            #     total_norm = total_norm ** (1. / 2)
            #     for p in self.network.parameters():
            #         p.grad = p.grad / total_norm
            
            if self.config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            if hasattr(config, "async_actor") and config.async_actor is True:
                with config.lock:
                    self.optimizer.step()
            else:
                self.optimizer.step()

        if config.use_target_network and self.total_steps / self.config.sgd_update_frequency % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        return report