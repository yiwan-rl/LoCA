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


class InterOptionDiffQLearningActor(BaseActor):
	def __init__(self, config):
		BaseActor.__init__(self, config)
		self.config = config
		self.start()
		self._option_start_state = None
		self._option_reward = None
		self._option_length = None
		self._option = None
		self._option_is_explorarory = None
		self._option_values = None
	
	def _transition(self):
		if self._state is None:
			self._state = self._task.reset()
		config = self.config
		
		if self._option is None:
			state_idx = np.where(self._state)[0][0]
			q_values = self._option_values[state_idx]
			if self._total_steps < config.exploration_steps or np.random.rand() < config.random_action_prob():
				self._option = np.random.randint(0, len(q_values))
				self._option_is_explorarory = True
			else:
				# self._option = np.random.choice(np.flatnonzero(np.isclose(q_values, q_values.max())))
				self._option = np.random.choice(np.flatnonzero(q_values == q_values.max()))
				# self._option = np.argmax(q_values)
				self._option_is_explorarory = False
			self._option_length = 0
			self._option_start_state = self._state
			self._option_reward = 0
		
		next_state, reward, done, info = self._task.step((self._option, False))
		self._option_length += 1
		self._option_reward += reward
		
		option_term = bool(np.random.choice([0, 1], p=[1 - info["option_term"], info["option_term"]]))
		
		if option_term:
			entry = [self._option_start_state, self._option, self._option_reward, next_state, self._option_length, self._option_is_explorarory]
			self._option = None
		else:
			entry = None
		self._total_steps += 1
		self._state = next_state
		return entry


class InterOptionDiffQLearning(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.actor = InterOptionDiffQLearningActor(config)
		self._option = None
		self.reward_rate = 0
		self.visitation = np.zeros((self.actor._task.state_dim, self.actor._task.action_dim))
		self.pred_option_length = np.ones_like(self.visitation)
		self.option_values = np.zeros_like(self.visitation)
		self.actor._option_values = self.option_values
		self.total_steps = 0
	
	def close(self):
		close_obj(self.actor)
	
	def eval_step(self, state, reward, done, info):
		if info is None:
			option_term = True
		else:
			option_term = bool(np.random.choice([0, 1], p=[1 - info["option_term"], info["option_term"]]))
		
		if option_term:
			state_idx = np.where(state)[0][0]
			q_values = self.option_values[state_idx]
			# self._option = np.random.choice(np.flatnonzero(np.isclose(q_values, q_values.max())))
			# self._option = np.argmax(q_values)
			self._option = np.random.choice(np.flatnonzero(q_values == q_values.max()))
			
		# if (self._total_steps % 1000) == 0:
		# print(np.where(state)[0][0])
		# 	q_values = self._network(config.state_normalizer(np.diag(np.ones(self._task.env.observation_space.shape[0]))))
		# 	print(self._total_steps, self._option_values[np.where(next_state)[0][0]])
		# 	print(self._task.env.render_policy_over_options_options_only(np.argmax(self._option_values, axis=1)))
		# print(self.actor._task.env.render_policy_over_options(np.argmax(self.option_values, axis=1), only_action=True))
		# print(self.actor._task.env.render_policy_over_options(np.argmax(self.option_values, axis=1), only_action=False))
		# print(self.reward_rate)
		return (self._option, False)
	
	def step(self):
		report = {}
		config = self.config
		transitions = self.actor.step()
		
		if transitions.__len__() != 0:
			state, option, reward, next_state, option_length, option_is_explorarory = transitions[0]
			report.setdefault('rewards', []).append(reward)
			state_idx = np.where(state)[0][0]
			self.visitation[state_idx, option] += 1
			self.pred_option_length[state_idx, option] += config.lr2 * (
					option_length - self.pred_option_length[state_idx, option]
			)
			pred_option_length = self.pred_option_length[state_idx, option]
			q_next = self.option_values[np.where(next_state)[0][0]].max(0)
			q = self.option_values[state_idx, option]
			td_error = reward - self.reward_rate * option_length + q_next - q
			self.option_values[state_idx, option] += config.lr * td_error / pred_option_length
			self.reward_rate += config.lr * config.diff_methods_eta * td_error / pred_option_length
			self.total_steps += option_length
				
		return report


class Gosavi2004(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.actor = InterOptionDiffQLearningActor(config)
		self._option = None
		self.reward_rate = 0
		self.exp_avg_option_reward = 0
		self.exp_avg_option_length = 0
		self.option_values = np.zeros((self.actor._task.state_dim, self.actor._task.action_dim))
		self.actor._option_values = self.option_values
		self.total_steps = 0
	
	def close(self):
		close_obj(self.actor)
	
	def eval_step(self, state, reward, done, info):
		if info is None:
			option_term = True
		else:
			option_term = bool(np.random.choice([0, 1], p=[1 - info["option_term"], info["option_term"]]))
		
		if option_term:
			state_idx = np.where(state)[0][0]
			q_values = self.option_values[state_idx]
			# self._option = np.random.choice(np.flatnonzero(np.isclose(q_values, q_values.max())))
			# self._option = np.argmax(q_values)
			self._option = np.random.choice(np.flatnonzero(q_values == q_values.max()))
		return (self._option, False)
	
	def step(self):
		report = {}
		config = self.config
		transitions = self.actor.step()
		if transitions.__len__() != 0:
			state, option, reward, next_state, option_length, option_is_explorarory = transitions[0]
			report.setdefault('rewards', []).append(reward)
			state_idx = np.where(state)[0][0]
			q_next = self.option_values[np.where(next_state)[0][0]].max(0)
			q = self.option_values[state_idx, option]
			td_error = reward - self.reward_rate * option_length + q_next - q
			self.option_values[state_idx, option] += config.lr * td_error
			if option_is_explorarory == False:
				self.exp_avg_option_reward += self.config.lr2 * (reward - self.exp_avg_option_reward)
				self.exp_avg_option_length += self.config.lr2 * (option_length - self.exp_avg_option_length)
				self.reward_rate = self.exp_avg_option_reward / self.exp_avg_option_length
			self.total_steps += option_length
		return report
	

class IntraOptionDiffQLearningActor(BaseActor):
	def __init__(self, config):
		BaseActor.__init__(self, config)
		self.config = config
		self.start()
		self._option = None
		self._option_values = None
	
	def _transition(self):
		if self._state is None:
			self._state = self._task.reset()
		config = self.config
		
		if self._option is None:
			if config.random_action_intra_option_exp:
				self._option = np.random.randint(0, 4)
			else:
				state_idx = np.where(self._state)[0][0]
				q_values = self._option_values[state_idx]
				if self._total_steps < config.exploration_steps or np.random.rand() < config.random_action_prob():
					self._option = np.random.randint(0, len(q_values))
					# while self._option == np.argmax(q_values):
					# 	self._option = np.random.randint(0, len(q_values))
				else:
					# self._option = np.random.choice(np.flatnonzero(np.isclose(q_values, q_values.max())))
					# self._option = np.argmax(q_values)
					self._option = np.random.choice(np.flatnonzero(q_values == q_values.max()))
		
		next_state, reward, done, info = self._task.step((self._option, config.random_action_intra_option_exp))

		entry = [self._state, self._option, reward, next_state, info]
		option_term = bool(np.random.choice([0, 1], p=[1 - info["option_term"], info["option_term"]]))
		
		if option_term is False and self.config.use_interruption:
			state_idx = np.where(next_state)[0][0]
			q_values = self._option_values[state_idx]
			# option = np.argmax(q_values)
			# if np.max(q_values) != q_values[self._option]:
			# 	option_term = True
			# option = np.argmax(q_values)
			if np.max(q_values) != q_values[self._option]:
				option_term = True
				
		if option_term:
			self._option = None

		self._total_steps += 1
		self._state = next_state
		# if self._total_steps > 125000:
		# 	print(self._task.env.render(), option_term)
		# print(self._total_steps, self._option_values[np.where(next_state)[0][0]])
		# if (self._total_steps % 1000) == 0:
		# # 	# print(self._task.env.render())
		# # 	q_values = self._network(config.state_normalizer(np.diag(np.ones(self._task.env.observation_space.shape[0]))))
		# print(self._total_steps, self._option_values[np.where(next_state)[0][0]])
		# print(self._task.env.render_policy_over_options_options_only(np.argmax(self._option_values, axis=1)))
		
		return entry


class IntraOptionDiffQLearning(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.actor = IntraOptionDiffQLearningActor(config)
		self.total_steps = 0
		self._option = None
		self.option_values = np.zeros((self.actor._task.state_dim, self.actor._task.action_dim))
		self.actor._option_values = self.option_values
		self.reward_rate = 0
		
	
	def close(self):
		close_obj(self.actor)
	
	def eval_step(self, state, reward, done, info):
		if info is None:
			option_term = True
		else:
			option_term = bool(np.random.choice([0, 1], p=[1 - info["option_term"], info["option_term"]]))
			
		if option_term is False and self.config.use_interruption:
			state_idx = np.where(state)[0][0]
			q_values = self.option_values[state_idx]
			# option = np.argmax(q_values)
			# if option != self._option:
			# 	option_term = True
			if np.max(q_values) != q_values[self._option]:
				option_term = True
		
		if option_term:
			state_idx = np.where(state)[0][0]
			q_values = self.option_values[state_idx]
			# self._option = np.random.choice(np.flatnonzero(np.isclose(q_values, q_values.max())))
			# self._option = np.argmax(q_values)
			self._option = np.random.choice(np.flatnonzero(q_values == q_values.max()))
			
		# print(self.actor._task.env.render_policy_over_options_options_only(np.argmax(self.option_values, axis=1),
		#                                                       only_action=True))
		# print(self.actor._task.env.render_policy_over_options_options_only(np.argmax(self.option_values, axis=1),
		#                                                       only_action=False))
		print(self.option_values[1], self.reward_rate)
		
		return (self._option, False)
	
	def step(self):
		report = {}
		config = self.config
		transitions = self.actor.step()
		if transitions.__len__() != 0:
			state, option, reward, next_state, info = transitions[0]
			report.setdefault('rewards', []).append(reward)
			q_next = self.option_values[np.where(next_state)[0][0]]
			max_q_next = q_next.max()
			options_term_probs = np.array(info["options_term_probs"])
			options_action_probs = np.array(info["options_action_probs"])
			if config.random_action_intra_option_exp:
				is_ratio = options_action_probs
			else:
				is_ratio = options_action_probs / options_action_probs[option]
			state_idx = np.where(state)[0][0]
			td_errors = reward - self.reward_rate + options_term_probs * max_q_next + (1 - options_term_probs) * q_next - self.option_values[state_idx]
			self.option_values[state_idx] += config.lr * is_ratio * td_errors
			self.reward_rate += config.diff_methods_eta * config.lr * np.sum(is_ratio * td_errors)
			self.total_steps += 1
		return report