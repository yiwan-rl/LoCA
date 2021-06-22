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
	
	def _transition(self):
		if self._state is None:
			self._state = self._task.reset()
		config = self.config
		
		if hasattr(config, "async_actor") and config.async_actor is True:
			with config.lock:
				q_values = self._network(config.state_normalizer(np.expand_dims(self._state, axis=0)))
		else:
			q_values = self._network(config.state_normalizer(np.expand_dims(self._state, axis=0)))
		
		q_values = to_np(q_values).flatten()
		if self._total_steps < config.exploration_steps or np.random.rand() < config.random_action_prob():
			action = np.random.randint(0, len(q_values))
		else:
			action = np.argmax(q_values)
		next_state, reward, done, info = self._task.step(action)
		
		if config.bootstrap_from_timeout is True and info['TimeLimit.truncated'] is True:
			done = False
		entry = [self._state, action, reward, next_state, int(done), info]
		self._total_steps += 1
		# if self._total_steps % 5000 == 0:
		# 	s = np.zeros_like(self._state)
		# 	s[0] = 1
		# 	print(self._network(self.config.state_normalizer(np.expand_dims(s, axis=0))))
		self._state = next_state
		return entry


class DiffQLearning(BaseAgent):
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
		
		self.reward_rate = nn.Linear(1, 1, bias=False)
		nn.init.constant_(self.reward_rate.weight.data, 0)
		
		parameters = list(self.network.parameters()) + list(self.reward_rate.parameters())
		
		self.optimizer = set_optimizer(parameters, config)
		
		self.actor.set_network(self.network)
		
		self.total_steps = 0
		self.batch_indices = range_tensor(self.replay.batch_size)
	
	def close(self):
		close_obj(self.replay)
		close_obj(self.actor)
	
	def eval_step(self, state, reward, done, info):
		self.config.state_normalizer.set_read_only()
		q = self.network(self.config.state_normalizer(np.expand_dims(state, axis=0)))
		action = to_np(q.argmax(-1))[0]
		self.config.state_normalizer.unset_read_only()
		return action
	
	def step(self):
		report = {}
		config = self.config
		transitions = self.actor.step()
		experiences = []
		for state, action, reward, next_state, done, info in transitions:
			report.setdefault('rewards', []).append(reward)
			# self.record_online_return(info)
			# ret = info['episodic_return']
			# if ret is not None:
			# 	report.setdefault('episodic_return', []).append(ret)
			# 	report.setdefault('episodic_length', []).append(info['episodic_length'])
			self.total_steps += 1
			reward = config.reward_normalizer(reward)
			experiences.append([state, action, reward, next_state, done])
		self.replay.feed_batch(experiences)
		
		if self.total_steps > self.config.exploration_steps:
			experiences = self.replay.sample()
			states, actions, rewards, next_states, terminals = experiences
			states = self.config.state_normalizer(states)
			next_states = self.config.state_normalizer(next_states)
			q_next = self.target_network(next_states).detach()
			if self.config.double_q:
				best_actions = torch.argmax(self.network(next_states), dim=-1)
				q_next = q_next[self.batch_indices, best_actions]
			else:
				q_next = q_next.max(1).values
			# terminals = tensor(terminals)
			rewards = tensor(rewards)
			# q_next = self.config.discount * q_next * (1 - terminals)
			reward_rate = self.reward_rate(torch.tensor([[1.]]))
			q_backup = q_next.detach() + rewards - reward_rate.detach()
			actions = tensor(actions).long()
			q = self.network(states)
			q = q[self.batch_indices, actions]
			reward_rate_backup = q_next.detach() + rewards - q.detach()
			q_loss = (q_backup - q).pow(2).mul(0.5).mean()
			reward_rate_loss = (reward_rate_backup - reward_rate).pow(2).mul(0.5).mean()
			
			report.setdefault('q_loss', []).append(to_np(q_loss))
			report.setdefault('q_values', []).append(to_np(q).mean())
			report.setdefault('r_bar_loss', []).append(to_np(reward_rate_loss))
			report.setdefault('r_bar_value', []).append(to_np(reward_rate))
			
			self.optimizer.zero_grad()
			(q_loss + self.config.differential_methods_eta * reward_rate_loss).backward()
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


class RVIQLearning(BaseAgent):
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
	
	def eval_step(self, state, reward, done, info):
		self.config.state_normalizer.set_read_only()
		q = self.network(self.config.state_normalizer(np.expand_dims(state, axis=0)))
		action = to_np(q.argmax(-1))[0]
		self.config.state_normalizer.unset_read_only()
		return action
	
	def step(self):
		report = {}
		config = self.config
		transitions = self.actor.step()
		experiences = []
		for state, action, reward, next_state, done, info in transitions:
			report.setdefault('rewards', []).append(reward)
			# self.record_online_return(info)
			# ret = info['episodic_return']
			# if ret is not None:
			# 	report.setdefault('episodic_return', []).append(ret)
			# 	report.setdefault('episodic_length', []).append(info['episodic_length'])
			self.total_steps += 1
			reward = config.reward_normalizer(reward)
			experiences.append([state, action, reward, next_state, done])
		self.replay.feed_batch(experiences)
		
		if self.total_steps > self.config.exploration_steps:
			experiences = self.replay.sample()
			states, actions, rewards, next_states, terminals = experiences
			states = self.config.state_normalizer(states)
			next_states = self.config.state_normalizer(next_states)
			q_next = self.target_network(next_states).detach()
			if self.config.double_q:
				best_actions = torch.argmax(self.network(next_states), dim=-1)
				q_next = q_next[self.batch_indices, best_actions]
			else:
				q_next = q_next.max(1).values
			# terminals = tensor(terminals)
			rewards = tensor(rewards)
			# q_next = self.config.discount * q_next * (1 - terminals)
			ref_fea_vec = np.diag(np.ones(states.shape[1]))
			# ref_fea_vec[0, 0] = 0
			# ref_fea_vec[0, 20] = 0
			# ref_fea_vec[0, 60] = 0
			# ref_fea_vec = np.random.randint(2, size=(1, states.shape[1])).astype(float)
			if self.config.RVIQLearningf == 'max':
				f = self.network(ref_fea_vec).max()
			elif self.config.RVIQLearningf == 'mean':
				f = self.network(ref_fea_vec).mean()
			else:
				raise NotImplementedError
			q_backup = q_next.detach() + rewards - f.detach()
			actions = tensor(actions).long()
			q = self.network(states)
			q = q[self.batch_indices, actions]
			q_loss = (q_backup - q).pow(2).mul(0.5).mean()
			
			report.setdefault('q_loss', []).append(to_np(q_loss))
			report.setdefault('q_values', []).append(to_np(q).mean())
			report.setdefault('r_bar_value', []).append(to_np(f))
			
			self.optimizer.zero_grad()
			q_loss.backward()
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