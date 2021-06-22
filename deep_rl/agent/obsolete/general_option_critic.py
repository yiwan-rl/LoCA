#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl.agent.BaseAgent import *
import matplotlib.pyplot as plt
import torch
from deep_rl.component.replay import Storage


class GeneralOptionCritic(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.task = config.task_fn(config.seed)
		self.network = config.network
		
		self.eval_network = config.network_fn()
		self.eval_network.load_state_dict(self.network.state_dict())
		
		self.mu_opt = config.optimizer_fn(self.network.mu_params)
		self.q_opt = config.optimizer_fn(self.network.q_params)
		self.pi_beta_opt = config.optimizer_fn(self.network.pi_beta_params)
		self.mine_opt = config.optimizer_fn(self.network.mine_params)
		
		self.switch_cost = 0
		self.total_steps = 0
		self.total_eps = 0
		self.roll_out_steps = 0
		self.storage = Storage(100)
		self.replay = config.replay_fn()
		
		self.prev_option_test = None
		self.gamma_power = None
		self.eps_steps = None
		self.total_mine_loss = None
		
		if self.config.task_name == "TwoRoomsThreeGoals":
			self.sample_path = None
			self.sample_path_count = 0
		
		# Initialize S_0
		self.state = self.task.reset()
		self.state = tensor([self.config.state_normalizer(self.state)])
		# Choose O_0
		self.option = self.network(self.state, 'mu')
		
		self.start()
	
	def eval_step(self, ep, state, reward, terminal, info):
		# self.config.state_normalizer.set_read_only()
		state = tensor([self.config.state_normalizer(state)])
		# self.config.state_normalizer.unset_read_only()
		
		if terminal is True:
			self.eval_network.load_state_dict(self.network.state_dict())
			if self.config.task_name == "TwoRoomsThreeGoals":
				state = self.config.eval_env.env.reset_test()
				# self.config.state_normalizer.set_read_only()
				state = tensor([self.config.state_normalizer(state)])
				# self.config.state_normalizer.unset_read_only()
			
			option_test = self.network(state, 'mu').detach()
			if self.config.task_name == "TwoRoomsThreeGoals" and ep == 0:
				self.print_sample_path()
				self.sample_path = {
					'cell': [], 'option': [], 'prev_option_term': [],
					'goal': self.config.eval_env.env.tocell[self.config.eval_env.env.goals[self.config.eval_env.env.goal_idx]]
				}
				self.sample_path['cell'].append(self.config.eval_env.env.current_cell)
				self.sample_path['prev_option_term'].append(1)
				self.sample_path['option'].append(option_test.detach())
		else:
			# terminate option
			beta_dist = self.network((state, self.prev_option_test), 'beta')
			prev_option_term = beta_dist.sample()[0, 0]
			# prev_option_term = True
			
			if prev_option_term:
				option_test = self.network(state, 'mu').detach()
			else:
				option_test = self.prev_option_test
			
			if self.config.task_name == "TwoRoomsThreeGoals" and ep == 0:
				self.sample_path['cell'].append(self.config.eval_env.env.current_cell)
				self.sample_path['prev_option_term'].append(prev_option_term)
				self.sample_path['option'].append(option_test.detach())
	
		pi_dist = self.network((state, option_test), 'pi')
		action = pi_dist.sample()[0]
		
		self.prev_option_test = option_test
		
		return to_np(action).item()
	
	def start(self):
		self.gamma_power = tensor([[1]])
		self.total_eps += 1
		self.eps_steps = 0
		self.total_mine_loss = 0
	
	def step(self):
		self.total_steps += 1
		self.eps_steps += 1
		self.roll_out_steps += 1
		
		eps_term, option_term, report = self.interact()

		if eps_term or self.roll_out_steps >= self.config.rollout_length: # option_term or
			self.train(eps_term, option_term)
			self.roll_out_steps = 0
			self.storage = Storage(100)
		
		# if self.total_eps % 50 == 0 and eps_term and self.config.rank == 0:
		# 	episodic_lengths = []
		# 	# self.temp_task = self.task
		# 	# self.temp_state = self.state
		# 	# self.temp_option = self.option
		# 	# self.task = self.config.eval_env
		# 	# state = tensor([self.config.state_normalizer(state)])
		# 	# self.prev_option = self.network(state, 'mu')
		# 	for ep in range(self.config.eval_episodes):
		# 		state = self.config.eval_env.reset()
		# 		reward = 0
		# 		eps_term = True
		# 		info = None
		# 		# while True:
		# 		# 	# state = tensor([self.config.state_normalizer(state)])
		# 		# 	action = self.interact_test(ep, state, reward, done, info)
		# 		# 	state, reward, done, info = self.task.step(action)
		# 		# 	if done:
		# 		# 		break
		# 		while True:
		# 			action = self.eval_step(ep, state, reward, eps_term, info)
		# 			# Obtain S_{t+1}, R_{t+1}, and whether or not the episode terminates
		# 			state, reward, eps_term, info = self.config.eval_env.step(action)
		# 			if eps_term:
		# 				break
		# 		# 	# action = self.eval_step(ep, state, reward, done, info)
		# 		# 	# state, reward, done, info = env.step(action)
		# 		# 	# ret = info['episodic_return']
		# 		# 	# len = info['episodic_length']
		# 		# 	# if done:
		# 		# 	# 	break
		# 		episodic_lengths.append(info['episodic_length'])
		# 	print(np.mean(episodic_lengths))
		# 	self.roll_out_steps = 0
		# 	self.storage = Storage(100)
		# 	# self.task = self.temp_task
		# 	# self.state = self.temp_state
		# 	# self.option = self.temp_option
		return report
	
	# def interact_test(self, ep, state, reward, done, info):
	# 	'''
	# 	interact with the environment with the current policy and options without updating any parameters
	# 	'''
	# 	# report = {}
	#
	# 	# given S_t, O_t, choose A_t
	#
	# 	pi_dist = self.network((state, self.prev_option), 'pi')
	# 	action = pi_dist.sample()[0]
	#
	# 	# Obtain S_{t+1}, R_{t+1}, and whether or not the episode terminates
	# 	next_state, reward, eps_term, info = self.task.step(to_np(action).item())
	#
	# 	next_state = tensor([self.config.state_normalizer(next_state)])
	#
	# 	# terminate option O_t in state S_{t+1}
	# 	beta_dist = self.network((next_state, self.prev_option), 'beta')
	# 	if eps_term:
	# 		option_term = True
	# 	else:
	# 		option_term = beta_dist.sample()[0, 0]
	#
	# 	# Choose O_{t+1}
	# 	if not eps_term:
	# 		if option_term:
	# 			next_option = self.network(next_state, 'mu')
	# 		else:
	# 			next_option = self.prev_option
	# 		self.prev_option = next_option
	# 	else:
	# 		# Choose O_0 for next episode
	# 		self.prev_option = self.network(state, 'mu')
	#
	# 	return to_np(action).item()
	#
	# 	# state = next_state
	# 	#
	# 	# if eps_term:
	# 	# 	# record info
	# 	# 	report.setdefault('episodic_return', []).append(info['episodic_return'])
	# 	# 	report.setdefault('episodic_length', []).append(info['episodic_length'])
	# 	# 	self.start()
	#
	# 	# return eps_term, option_term, report
	
	def interact_test(self, ep, state, reward, eps_term, info):
		'''
		interact with the environment with the current policy and options without updating any parameters
		'''
		state = tensor([self.config.state_normalizer(state)])
		
		# Choose O_{t+1}
		if not eps_term:
			# terminate option O_t in state S_{t+1}
			beta_dist = self.network((state, self.prev_option), 'beta')
			
			option_term = beta_dist.sample()[0, 0]
			if option_term:
				next_option = self.network(state, 'mu')
			else:
				next_option = self.prev_option
			# self.prev_option = next_option
		else:
			# Choose O_0 for next episode
			# self.prev_option = self.network(state, 'mu')
			next_option = self.network(state, 'mu')

		# given S_t, O_t, choose A_t

		pi_dist = self.network((state, next_option), 'pi')
		action = pi_dist.sample()[0]
		self.prev_option = next_option

		# self.state = next_state

		# if eps_term:
		# 	# record info
		# 	# report.setdefault('episodic_return', []).append(info['episodic_return'])
		# 	# report.setdefault('episodic_length', []).append(info['episodic_length'])
		# 	self.start()

		return to_np(action).item()
	
	# def interact_test(self, state):
	# 	'''
	# 	interact with the environment with the current policy and options without updating any parameters
	# 	'''
	# 	# report = {}
	#
	# 	# given S_t, O_t, choose A_t
	#
	# 	pi_dist = self.network((state, self.prev_option), 'pi')
	# 	action = pi_dist.sample()[0]
	#
	# 	# Obtain S_{t+1}, R_{t+1}, and whether or not the episode terminates
	# 	next_state, reward, eps_term, info = self.task.step(to_np(action).item())
	#
	# 	next_state = tensor([self.config.state_normalizer(next_state)])
	#
	# 	# terminate option O_t in state S_{t+1}
	# 	beta_dist = self.network((next_state, self.prev_option), 'beta')
	# 	if eps_term:
	# 		option_term = True
	# 	else:
	# 		option_term = beta_dist.sample()[0, 0]
	#
	# 	# Choose O_{t+1}
	# 	if not eps_term:
	# 		if option_term:
	# 			next_option = self.network(next_state, 'mu')
	# 		else:
	# 			next_option = self.prev_option
	# 		self.prev_option = next_option
	# 	else:
	# 		# Choose O_0 for next episode
	# 		self.prev_option = self.network(next_state, 'mu')
	#
	# 	# self.state = next_state
	#
	# 	# if eps_term:
	# 	# 	# record info
	# 	# 	# report.setdefault('episodic_return', []).append(info['episodic_return'])
	# 	# 	# report.setdefault('episodic_length', []).append(info['episodic_length'])
	# 	# 	self.start()
	#
	# 	return eps_term, option_term, info, next_state
	
	def interact(self):
		'''
		interact with the environment with the current policy and options without updating any paramaters
		'''
		report = {}
		
		# Store S_t, O_t in the replay buffer
		self.replay.feed([self.state.data.numpy()[0], self.option.data.numpy()[0]])
		
		# given S_t, O_t, choose A_t
		# self.option = self.network(self.state, 'mu')
		
		pi_dist = self.network((self.state, self.option), 'pi')
		action = pi_dist.sample()[0]
		log_pi = pi_dist.log_prob(action).unsqueeze(0)
		pi_entropy = pi_dist.entropy().unsqueeze(0)
		
		# Obtain S_{t+1}, R_{t+1}, and whether or not the episode terminates
		next_state, reward, eps_term, info = self.task.step(to_np(action).item())
		
		next_state = tensor([self.config.state_normalizer(next_state)])
		reward = tensor([[reward - self.switch_cost]])
		
		# terminate option O_t in state S_{t+1}
		beta_dist = self.network((next_state, self.option), 'beta')
		next_beta = beta_dist.probs
		beta_entropy = beta_dist.entropy()
		if eps_term:
			option_term = True
		else:
			option_term = beta_dist.sample()[0, 0]
		
		# next_beta = tensor([1.])
		# beta_entropy = tensor([0.])
		# option_term = True
		
		# Store (S_t, O_t, R_{t+1}, log \pi(A_t|S_t, O_t), \beta(S_{t+1}, O_t)) for training
		self.storage.add({
			'next_beta': next_beta, 'beta_entropy': beta_entropy,
			'log_pi': log_pi, 'pi_entropy': pi_entropy,
			'reward': reward, 'option': self.option,
			'state': self.state, 'gamma_power': self.gamma_power
		})
		
		# Choose O_{t+1}
		if not eps_term:
			if option_term:
				next_option = self.network(next_state, 'mu')
				self.switch_cost = self.config.switch_cost
			else:
				next_option = self.option
				self.switch_cost = 0
			self.option = next_option
		else:
			# Choose O_0 for next episode
			self.option = self.network(next_state, 'mu')
			self.switch_cost = 0
		
		self.state = next_state
	
		# self.gamma_power = self.gamma_power * config.discount
		
		# pi_dist = self.network((self.state, self.option), 'pi')
		# action = pi_dist.sample()[0]
		
		if eps_term:
			# record info
			report.setdefault('episodic_return', []).append(info['episodic_return'])
			report.setdefault('episodic_length', []).append(info['episodic_length'])
			report.setdefault('mine_loss', []).append(self.total_mine_loss)
			self.start()
	
		return eps_term, option_term, report
	
	def train(self, eps_term, option_term):
		'''
		Discover options and policy-over-them
		:param eps_term: whether or not the episode terminates at current time step
		:param option_term: whether or not the previous option terminates at current state
		:return: None
		'''
		# Extract (S_t, O_t, R_{t+1}, log \pi(A_t|S_t, O_t), \beta(S_{t+1}, O_t)) for training
		states, options, next_beta, beta_entropy, log_pi, pi_entropy, gamma_power = self.storage.cat(
			['state', 'option', 'next_beta', 'beta_entropy', 'log_pi', 'pi_entropy', 'gamma_power']
		)
		next_states = torch.cat((states[1:], self.state), dim=0)
		mu_next_states = self.network(next_states, 'mu')
		
		q = self.network(
			(
				torch.cat((states, next_states, next_states), dim=0),
				torch.cat((options.detach(), options.detach(), mu_next_states), dim=0)
			), 'q'
		)
		
		q_s_o = q[:self.roll_out_steps]
		q_next_s_o = q[self.roll_out_steps:self.roll_out_steps * 2]
		q_next_s_mu_next_s = q[self.roll_out_steps * 2:]
		
		# compute bootstrapping value
		if eps_term:
			ret = 0
		else:
			ret = q_next_s_mu_next_s[-1]
		
		rets = []
		for i in reversed(range(self.roll_out_steps)):
			ret = self.storage.reward[i] + self.config.discount * ret
			rets.append(ret)
		rets.reverse()
		rets = torch.cat(rets, dim=0)
		
		# # train policy over options
		# if eps_term:
		# 	next_beta[-1][0] = 1
		# mu_loss = - (gamma_power * next_beta.detach() * q_next_s_mu_next_s).mean()
		# self.network.zero_grad()
		# (self.config.mu_weight * mu_loss).backward(retain_graph=True)
		# self.mu_opt.step()
		
		# train critic
		q_loss = ((rets.detach() - q_s_o).pow(2).mul(0.5)).mean()
		self.network.zero_grad()
		(self.config.q_weight * q_loss).backward()
		self.q_opt.step()
		
		# train option and policy over options
		pi_loss = - (gamma_power * log_pi * (rets - q_s_o).detach()).mean()
		pi_entropy_loss = - (gamma_power * pi_entropy).mean()
		beta_loss = self.config.discount * gamma_power * next_beta * (q_next_s_o - q_next_s_mu_next_s + self.config.switch_cost).detach()
		beta_entropy_loss = - (gamma_power * beta_entropy)
		if eps_term:
			if beta_loss.shape[0] > 1:
				beta_loss = beta_loss[:-1].mean()
				beta_entropy_loss = beta_entropy_loss[:-1].mean()
			else:
				beta_loss = 0
				beta_entropy_loss = 0
		else:
			beta_loss = beta_loss.mean()
			beta_entropy_loss = beta_entropy_loss.mean()
		
		reg_orthogonal_loss = self.get_orthogonal_loss()
		reg_l1_loss = self.get_l1_loss()
		reg_mine_loss = self.get_mine_loss()
		
		self.network.zero_grad()
		(
				self.config.pi_beta_weight * (pi_loss + beta_loss)
				+ self.config.pi_entropy_weight * pi_entropy_loss
				+ self.config.beta_entropy_weight * beta_entropy_loss
				+ self.config.reg_orthogonal_weight * reg_orthogonal_loss
				+ self.config.reg_l1_weight * reg_l1_loss
				+ self.config.reg_mine_weight * reg_mine_loss
		).backward(retain_graph=not eps_term)
		self.pi_beta_opt.step()
		self.mu_opt.step()
		self.mine_opt.step()
	
		# if self.total_steps < 40000:
		# 	self.pi_beta_opt.step()
		
	def print_sample_path(self):
		if self.sample_path is None:
			return
		option_feature_dim = self.config.pi_option_dim + self.config.beta_option_dim
		# num_terms_to_print = option_feature_dim
		num_terms_to_print = 4
		curr_term = 0
		cell_map = self.task.env.occupancy.astype('float64')
		cell_map[self.sample_path['goal'][0], self.sample_path['goal'][1]] = 1
		cell_map[self.sample_path['cell'][0][0], self.sample_path['cell'][0][1]] = 0.3
		
		fig, axs = plt.subplots(5, num_terms_to_print, figsize=(20, 10))
		
		state_list = []
		for i in range(self.task.env.height):
			for j in range(self.task.env.width):
				state_list.append(
					self.task.env.to_obs((i, j), (self.sample_path['goal'][0], self.sample_path['goal'][1])))
		states = tensor(state_list)
		# states_features = self.network(states, 'state features')
		
		for idx in range(1, len(self.sample_path['prev_option_term']) + 1):
			if idx != len(self.sample_path['prev_option_term']):
				if self.sample_path['prev_option_term'][idx] == 0:
					cell_map[self.sample_path['cell'][idx][0], self.sample_path['cell'][idx][1]] = 0.3
					continue
					
				# sample path
				cell_map[self.sample_path['cell'][idx][0], self.sample_path['cell'][idx][1]] = 0.6
			
			im = axs[0, curr_term].imshow(cell_map, cmap='Blues')
			fig.colorbar(im, ax=axs[0, curr_term])
			
			# policy and termination under different options
				
			pi_y = [self.task.env.occupancy.astype('float64') for _ in range(self.task.env.action_space.n)]
			pi_x = [self.task.env.occupancy.astype('float64') for _ in range(self.task.env.action_space.n)]
			pi_map = self.task.env.occupancy.astype('str')
			beta_map = self.task.env.occupancy.astype('float64')
			q_map = self.task.env.occupancy.astype('float64')
			
			# print('option ', self.sample_path['option'][idx])
			pi_dist = self.network((states, self.sample_path['option'][idx-1].expand(self.task.env.height * self.task.env.width, -1)), 'pi')
			beta_dist = self.network((states, self.sample_path['option'][idx-1].expand(self.task.env.height * self.task.env.width, -1)), 'beta')
			q = self.network((states, self.sample_path['option'][idx-1].expand(self.task.env.height * self.task.env.width, -1)), 'q')
			for i in range(self.task.env.height):
				for j in range(self.task.env.width):
					if self.task.env.occupancy[(i, j)] == 0:
						beta_map[i, j] = beta_dist.probs[i * self.task.env.width + j, 0].data.numpy()
						q_map[i, j] = q[i * self.task.env.width + j, 0].data.numpy()
						for a in range(self.task.env.action_space.n):
							y, x = self.task.env.directions[a]
							pi_y[a][i, j] = -y * pi_dist.probs[i * self.task.env.width + j, a].data.numpy()
							pi_x[a][i, j] = x * pi_dist.probs[i * self.task.env.width + j, a].data.numpy()
						directions = ['^', "v", '<', '>']
						pi_map[i, j] = directions[torch.argmax(pi_dist.probs[i * self.task.env.width + j]).squeeze().data.numpy()]
					else:
						for a in range(self.task.env.action_space.n):
							pi_y[a][i, j] = 0
							pi_x[a][i, j] = 0

			Y = np.arange(0, self.task.env.height, 1)
			X = np.arange(0, self.task.env.width, 1)
			
			for a in range(self.task.env.action_space.n):
				axs[0, curr_term].quiver(X, Y, pi_x[a], pi_y[a], scale=15.0)
			
			axs[0, curr_term].axis('off')
			
			im = axs[1, curr_term].imshow(beta_map, cmap='Blues')
			axs[1, curr_term].axis('off')
			fig.colorbar(im, ax=axs[1, curr_term])
			
			im = axs[2, curr_term].imshow(q_map, cmap='Blues')
			axs[2, curr_term].axis('off')
			fig.colorbar(im, ax=axs[2, curr_term])
			
			curr_term += 1
			if curr_term == num_terms_to_print:
				break
				
		# option features
		
		option = self.network(states, 'mu')
		option = option.view(self.task.env.height, self.task.env.width, -1).data.numpy()
		
		cell_map = self.task.env.occupancy.astype('float64')
		cell_map[self.sample_path['goal'][0], self.sample_path['goal'][1]] = 1
		cell_map[self.sample_path['cell'][0][0], self.sample_path['cell'][0][1]] = 0.3
		
		for option_feature in range(num_terms_to_print):
			if option_feature == option_feature_dim:
				break
				
			ax = axs[3, option_feature]
			im = ax.imshow(option[:, :, option_feature], cmap='Blues')
			ax.axis('off')
			fig.colorbar(im, ax=ax)
			# print(option_maps[:, :, option_feature].astype("int"))
			
			im = axs[4, option_feature].imshow(cell_map, cmap='Blues')
			# ax.axis('off')
			fig.colorbar(im, ax=axs[4, option_feature])
			pi_y = [self.task.env.occupancy.astype('float64') for _ in range(self.task.env.action_space.n)]
			pi_x = [self.task.env.occupancy.astype('float64') for _ in range(self.task.env.action_space.n)]
			
			# print('option ', self.sample_path['option'][idx])
			temp = torch.zeros((states.shape[0], option_feature_dim))
			temp[:, option_feature] = 1
			pi_dist = self.network((states, temp), 'pi')
			for i in range(self.task.env.height):
				for j in range(self.task.env.width):
					if self.task.env.occupancy[(i, j)] == 0:
						for a in range(self.task.env.action_space.n):
							y, x = self.task.env.directions[a]
							pi_y[a][i, j] = -y * pi_dist.probs[i * self.task.env.width + j, a].data.numpy()
							pi_x[a][i, j] = x * pi_dist.probs[i * self.task.env.width + j, a].data.numpy()
					else:
						for a in range(self.task.env.action_space.n):
							pi_y[a][i, j] = 0
							pi_x[a][i, j] = 0
			
			Y = np.arange(0, self.task.env.height, 1)
			X = np.arange(0, self.task.env.width, 1)
			
			for a in range(self.task.env.action_space.n):
				axs[4, option_feature].quiver(X, Y, pi_x[a], pi_y[a], scale=15.0)
			
			axs[4, option_feature].axis('off')

		# im.set_clim(0, 1)

		plt.savefig("termination_%d.pdf" % self.sample_path_count)
		self.sample_path_count += 1
		# plt.show(block=False)
		# plt.pause(1)
		plt.close()
		
	def get_kl_loss(self):
		# kl divergence

		ref_pi_dist = self.network((next_states, options.detach()), 'pi')
		new_pi_dist = self.network((next_states, options_in_next_states.detach()), 'pi')
		ref_pi = ref_pi_dist.probs
		batch_kl_loss = (ref_pi * (torch.log_softmax(ref_pi_dist.logits, dim=1) - torch.log_softmax(new_pi_dist.logits, dim=1))).sum(dim=1)
		kl_loss = torch.sum((options - options_in_next_states).abs().sum(dim=1).squeeze().detach() * batch_kl_loss)
	
	def get_l1_loss(self):
		reg_l1_loss = 0
		for param in self.network.mu_params + self.network.pi_params + self.network.beta_params:
			param_loss = torch.sum(torch.abs(param))
			reg_l1_loss += param_loss
		return reg_l1_loss
	
	# reg_l1_loss = 0
	# for param in list(self.network.mu_body.parameters()) + list(self.network.fc_mu.parameters()) + \
	#              list(self.network.pi_body.parameters()) + list(self.network.beta_body.parameters()):
	# 	param_loss = torch.sum(torch.abs(param))
	# 	reg_l1_loss += param_loss
	# 	self.total_reg_l1_loss += param_loss.data.numpy()
	# return reg_l1_loss
	
	def get_sparsity_loss(self):
		reg_sparsity_loss = 0
		batch_size = 32
		if self.config.reg_sparsity_weight != 0 and self.replay.size() > 2 * batch_size:
			random_states = self.replay.sample(batch_size)[0]
			random_states = tensor([self.config.state_normalizer(random_states)]).squeeze()
			state_features = self.network(random_states, 'pi phi')
			reg_sparsity_loss = state_features.abs().mean()
		return reg_sparsity_loss
	
	def get_orthogonal_loss(self):
		reg_orthogonal_loss = 0
		batch_size = 32
		if self.config.reg_orthogonal_weight != 0 and self.replay.size() > 2 * batch_size:
			random_states = self.replay.sample(2 * batch_size)[0]
			option = self.network(tensor(self.config.state_normalizer(random_states)), 'mu')
			a = option[:batch_size]
			b = option[batch_size:]
			reg_orthogonal_loss += torch.bmm(a.view(batch_size, 1, -1), b.view(batch_size, -1, 1)).pow(2).view(batch_size) - \
			                       torch.sum(a ** 2, 1) - \
			                       torch.sum(b ** 2, 1) + \
			                       self.config.pi_option_dim + self.config.beta_option_dim
			reg_orthogonal_loss = reg_orthogonal_loss.mean()
			
			# self.total_reg_orthogonal_loss += reg_orthogonal_loss.data.numpy()
		return reg_orthogonal_loss
	
	def get_decorrelated_loss(self):
		reg_decorrelated_loss = 0
		batch_size = 64
		if self.config.reg_decorrelated_weight != 0 and self.replay.size() > batch_size * 6:
			random_states = self.replay.sample(6 * batch_size)[0]
			option = self.network(np.expand_dims(self.config.state_normalizer(random_states), axis=0), 'mu')[0]
			a = option[:batch_size]
			b = option[batch_size:2 * batch_size]
			c = option[2 * batch_size:3 * batch_size]
			d = option[3 * batch_size:4 * batch_size]
			e = option[4 * batch_size:5 * batch_size]
			f = option[5 * batch_size:6 * batch_size]
			ab = torch.bmm(a.view(batch_size, 1, self.config.option_feature_dim),
			               b.view(batch_size, self.config.option_feature_dim, 1)).pow(2).view(batch_size)
			ac = torch.bmm(a.view(batch_size, 1, self.config.option_feature_dim),
			               c.view(batch_size, self.config.option_feature_dim, 1)).pow(2).view(batch_size)
			ad = torch.bmm(a.view(batch_size, 1, self.config.option_feature_dim),
			               d.view(batch_size, self.config.option_feature_dim, 1)).pow(2).view(batch_size)
			be = torch.bmm(b.view(batch_size, 1, self.config.option_feature_dim),
			               e.view(batch_size, self.config.option_feature_dim, 1)).pow(2).view(batch_size)
			bf = torch.bmm(b.view(batch_size, 1, self.config.option_feature_dim),
			               f.view(batch_size, self.config.option_feature_dim, 1)).pow(2).view(batch_size)
			ce = torch.bmm(c.view(batch_size, 1, self.config.option_feature_dim),
			               e.view(batch_size, self.config.option_feature_dim, 1)).pow(2).view(batch_size)
			df = torch.bmm(d.view(batch_size, 1, self.config.option_feature_dim),
			               f.view(batch_size, self.config.option_feature_dim, 1)).pow(2).view(batch_size)
			reg_decorrelated_loss = (ab * ab - ac * ad - be * bf + ce * df).mean()
			self.total_reg_decorrelated_loss += reg_decorrelated_loss.data.numpy()
		return reg_decorrelated_loss
	
	def get_mine_loss(self):
		'''
		Maximize the mutual information between options and the corresponding intra-option policies.
		:return: the loss to be optimized
		'''
		reg_mine_loss = 0
		if self.config.reg_mine_weight != 0:
			states1, options1 = self.replay.sample(64)
			states2, options2 = self.replay.sample(64)
			states1 = tensor(states1)
			options1 = tensor(options1)
			options2 = tensor(options2)
			dist = self.network((states1, options1), 'pi')
			pi = dist.probs
			pred_xy = self.network((options1, pi), 'mine')
			pred_x_y = self.network((options2, pi), 'mine')
			reg_mine_loss = - (torch.mean(pred_xy) - torch.logsumexp(pred_x_y, dim=0)) - log(64)
			self.total_mine_loss = self.total_mine_loss + reg_mine_loss.data.data.numpy()
		return reg_mine_loss