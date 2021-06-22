#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl.agent.BaseAgent import *
import matplotlib.pyplot as plt
import torch


class DeterministicOptionCritic(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.task = config.task_fn(config.seed)
		self.network = config.network
		
		self.mu_opt = config.optimizer_fn(self.network.mu_params)
		self.q_opt = config.optimizer_fn(self.network.q_params)
		self.pi_opt = config.optimizer_fn(self.network.pi_parameters)
		self.beta_opt = config.optimizer_fn(self.network.beta_parameters)
		self.mine_opt = config.optimizer_fn(self.network.mine_parameters)
		
		self.total_steps = 0
		
		self.gamma_power = None
		
		self.state = self.task.reset()
		self.option = None
		self.q = None
		
		self.switch_cost = 0
		
		if config.task_name == "TwoRoomsThreeGoals":
			self.termination_map = self.task.env.occupancy.astype('float64')
		
		if config.reg_orthogonal_weight != 0 or config.reg_decorrelated_weight != 0 or config.reg_mine_weight != 0:
			self.replay = config.replay_fn()
			
		# DEBUG
		self.total_eps = 0
		
		self.start()
	
	def step(self):
		config = self.config
			
		report = {}
		self.total_steps += 1
		self.eps_steps += 1
		
		mu_loss = - config.mu_entropy_weight * self.mu_entropy
		
		dist = self.network((np.expand_dims(config.state_normalizer(self.state), axis=0), self.option), 'pi')
		action = dist.sample()[0]
		log_pi = dist.log_prob(action)[0]
		pi_entropy = dist.entropy()[0]
		
		next_state, reward, terminal, info = self.task.step(to_np(action).item())
		
		if config.reg_orthogonal_weight != 0 or config.reg_decorrelated_weight != 0 or config.reg_mine_weight != 0:
			self.replay.feed([self.state])
		
		reward -= self.switch_cost
		
		if not terminal:
			
			dist = self.network(
				(
					np.expand_dims(config.state_normalizer(next_state), axis=0),
					self.option
				),
				'beta'
			)
			
			# terminate option
			beta_next_state = dist.probs[0, 0]
			option_term = dist.sample()[0, 0]
			beta_entropy = dist.entropy()[0, 0]
			
			# choose option
			mu_next_state, self.mu_entropy = self.network(np.expand_dims(self.config.state_normalizer(next_state), axis=0), 'mu')
			
			q = self.network((np.expand_dims(config.state_normalizer(next_state), axis=0).repeat(2, axis=0), torch.cat((self.option, mu_next_state), dim=0).detach()), 'q')
			
			q_next_s_o = q[0]
			q_next_s_mu_next_s = q[1]
			
			expected_next_q = (1 - beta_next_state) * q_next_s_o + beta_next_state * q_next_s_mu_next_s
			
			expected_td_error = reward + config.discount * expected_next_q.detach() - self.q
			
			beta_adv = q_next_s_o - q_next_s_mu_next_s + config.switch_cost
			
			beta_loss = self.gamma_power * config.discount * beta_next_state * beta_adv.detach() - config.beta_entropy_weight * beta_entropy
			
			self.total_beta_loss += beta_loss.data.numpy()
			
			if option_term:
				next_option = mu_next_state
				next_q = q_next_s_mu_next_s
				self.switch_cost = config.switch_cost
				self.switch_times += 1
				if config.task_name == "TwoRoomsThreeGoals":
					self.sample_path['option_term'].append(1)
			else:
				next_option = self.option
				next_q = q_next_s_o
				self.switch_cost = 0
				if config.task_name == "TwoRoomsThreeGoals":
					self.sample_path['option_term'].append(0)
			if config.task_name == "TwoRoomsThreeGoals":
				self.sample_path['cell'].append(self.task.env.current_cell)
				self.sample_path['option'].append(next_option.detach())
				self.sample_path['q'].append(next_q.data.numpy())
		else:
			expected_td_error = reward - self.q
			beta_loss = 0
			option_term = True

		q_loss = expected_td_error.pow(2).mul(0.5)
		pi_loss = - self.gamma_power * log_pi * expected_td_error.detach() - config.pi_entropy_weight * pi_entropy
		reg_loss = config.reg_l1_weight * self.get_l1_loss() + \
		           config.reg_orthogonal_weight * self.get_orthogonal_loss() + \
		           config.reg_decorrelated_weight * self.get_decorrelated_loss() + \
		           config.reg_mine_weight * self.get_mine_loss()
		
		# DEBUG and PRINT
		self.total_q_loss += q_loss.data.numpy()
		self.total_pi_loss += pi_loss.data.numpy()
		
		# optimize
		self.network.zero_grad()
		(
			config.mu_weight * mu_loss +
			config.q_weight * q_loss +
			config.pi_beta_weight * (pi_loss + beta_loss) +
			reg_loss
		).backward(retain_graph=(not option_term))
		
		if config.gradient_clip is not None:
			torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
		
		self.mu_opt.step()
		self.q_opt.step()
		self.pi_opt.step()
		self.beta_opt.step()
		
		if config.reg_mine_weight != 0:
			self.mine_opt.step()
		
		# post-processing
		self.state = next_state
		
		# # print info
		# if config.task_name == "TwoRoomsThreeGoals" and self.total_steps % 5000 == 0 and config.rank == 0:
		# 	self.print()
		
		if terminal:
			# record info
			report.setdefault('episodic_return', []).append(info['episodic_return'])
			report.setdefault('episodic_length', []).append(info['episodic_length'])
			report.setdefault('pi_loss', []).append(self.total_pi_loss / self.eps_steps)
			report.setdefault('beta_loss', []).append(self.total_beta_loss / self.eps_steps)
			report.setdefault('q_loss', []).append(self.total_q_loss / self.eps_steps)
			report.setdefault('reg_l1_loss', []).append(self.total_reg_l1_loss / self.eps_steps)
			report.setdefault('reg_orthogonal_loss', []).append(self.total_reg_orthogonal_loss / self.eps_steps)
			report.setdefault('reg_decorrelated_loss', []).append(self.total_reg_decorrelated_loss / self.eps_steps)
			report.setdefault('reg_mine_loss', []).append(self.total_reg_mine_loss / self.eps_steps)
			
			# print term info
			if config.task_name == "TwoRoomsThreeGoals" and self.total_eps % config.print_eps_interval == 0 and config.rank == 0:
				self.print_sample_path()
			self.start()
		else:
			self.option = next_option
			self.q = next_q
			# self.gamma_power *= config.discount
		
		return report
	
	def start(self):
		self.gamma_power = 1
		
		self.option, self.mu_entropy = self.network(np.expand_dims(self.config.state_normalizer(self.state), axis=0), 'mu')
		
		self.q = self.network((np.expand_dims(self.config.state_normalizer(self.state), axis=0), self.option.detach()), 'q')[0]
		
		# DEBUG and PRINT
		self.sample_path = {'q': [], 'cell': [], 'option': [], 'option_term': [], 'goal': self.task.env.tocell[self.task.env.goals[self.task.env.goal_idx]]}
		self.sample_path['cell'].append(self.task.env.current_cell)
		self.sample_path['option'].append(self.option.detach())
		self.sample_path['option_term'].append(1)
		self.sample_path['q'].append(self.q.data.numpy())
		
		self.switch_times = 1
		
		self.total_pi_loss = 0
		self.total_beta_loss = 0
		self.total_q_loss = 0
		self.total_reg_l1_loss = 0
		self.total_reg_orthogonal_loss = 0
		self.total_reg_decorrelated_loss = 0
		self.total_reg_mine_loss = 0
		
		self.total_eps += 1
		self.eps_steps = 0
		
	def sample_noise(self):
		return np.random.normal(size=(1, self.config.noise_dim))
	
	def print(self):
		# num_goals = self.task.env.goals.__len__()
		num_goals = 1
		
		option_features_maps = [
			[self.task.env.occupancy.astype('float64') for _ in range(num_goals)]
			for _ in range(self.config.option_feature_dim)
		]
		
		for goal_idx in range(num_goals):
			goal_cell = self.task.env.tocell[self.task.env.goals[goal_idx]]
			for i in range(self.task.env.height):
				for j in range(self.task.env.width):
					mu, _ = self.network(
						np.expand_dims(
							self.config.state_normalizer(self.task.env.to_obs((i, j), goal_cell)), axis=0
						), 'mu'
					)[0]
					for option in range(self.config.option_feature_dim):
						option_features_maps[option][goal_idx][i, j] = mu[option]
					
					if i == goal_cell[0] and j == goal_cell[1]:
						for option in range(self.config.option_feature_dim):
							option_features_maps[option][goal_idx][i, j] = 1
		
		fig, axs = plt.subplots(2, self.config.option_feature_dim * num_goals, figsize=(20, 10))
		
		ax = axs[0, 0]
		im = ax.imshow(self.termination_map, cmap='Blues')
		ax.axis('off')
		fig.colorbar(im, ax=ax)
		self.termination_map = self.task.env.occupancy.astype('float64')
		# im.set_clim(0, 1)
		
		ax = axs[0, 1]
		im = ax.imshow(self.sample_path_to_print, cmap='Blues')
		ax.axis('off')
		fig.colorbar(im, ax=ax)
		
		ax = axs[0, 2]
		im = ax.imshow(self.sample_path_q_to_print, cmap='Blues')
		ax.axis('off')
		fig.colorbar(im, ax=ax)
		
		for i in range(self.config.option_feature_dim):
			for j in range(num_goals):
				# print(mu_maps[i][j])
				ax = axs[1, i * num_goals + j]
				im = ax.imshow(option_features_maps[i][j], cmap='Blues')
				ax.axis('off')
				fig.colorbar(im, ax=ax)
				# im.set_clim(0, 1)
		
		plt.savefig("test_%d.pdf" % self.config.sweep_id)
		# plt.show(block=False)
		# plt.pause(1)
		plt.close()
		
	def print_sample_path(self):
		num_terms_to_print = 4
		curr_term = 0
		cell_map = self.task.env.occupancy.astype('float64')
		cell_map[self.sample_path['goal'][0], self.sample_path['goal'][1]] = 1
		
		fig, axs = plt.subplots(3, num_terms_to_print, figsize=(20, 10))
		
		for idx in range(len(self.sample_path['option_term'])):
			if self.sample_path['option_term'][idx] == 0:
				cell_map[self.sample_path['cell'][idx][0], self.sample_path['cell'][idx][1]] = 0.3
				continue
				
			# sample path
			
			cell_map[self.sample_path['cell'][idx][0], self.sample_path['cell'][idx][1]] = 0.6
			
			if num_terms_to_print == 1:
				im = axs[0].imshow(cell_map, cmap='Blues')
				fig.colorbar(im, ax=axs[0])
			else:
				im = axs[0, curr_term].imshow(cell_map, cmap='Blues')
				fig.colorbar(im, ax=axs[0, curr_term])
			
			# policy and termination under different options
				
			pi_y = [self.task.env.occupancy.astype('float64') for _ in range(self.task.env.action_space.n)]
			pi_x = [self.task.env.occupancy.astype('float64') for _ in range(self.task.env.action_space.n)]
			beta_map = self.task.env.occupancy.astype('float64')
			
			state_list = []
			for i in range(self.task.env.height):
				for j in range(self.task.env.width):
					state_list.append(self.task.env.to_obs((i, j), (self.sample_path['goal'][0], self.sample_path['goal'][1])))
			pi_dist = self.network((np.array(state_list), self.sample_path['option'][idx].expand(len(state_list), -1)), 'pi')
			beta_dist = self.network((np.array(state_list), self.sample_path['option'][idx].expand(len(state_list), -1)), 'beta')
			for i in range(self.task.env.height):
				for j in range(self.task.env.width):
					if self.task.env.occupancy[(i, j)] == 0:
						beta_map[i, j] = beta_dist.probs[i * self.task.env.width + j, 0].data.numpy()
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
			
			if num_terms_to_print == 1:
				for a in range(self.task.env.action_space.n):
					axs[0].quiver(X, Y, pi_x[a], pi_y[a], scale=15.0)
				
				axs[0].axis('off')
				
				im = axs[1].imshow(beta_map, cmap='Blues')
				axs[1].axis('off')
				fig.colorbar(im, ax=axs[1])
			else:
				for a in range(self.task.env.action_space.n):
					axs[0, curr_term].quiver(X, Y, pi_x[a], pi_y[a], scale=15.0)
				
				axs[0, curr_term].axis('off')
				
				im = axs[1, curr_term].imshow(beta_map, cmap='Blues')
				axs[1, curr_term].axis('off')
				fig.colorbar(im, ax=axs[1, curr_term])
			
			curr_term += 1
			if curr_term == num_terms_to_print:
				break
				
		# option features
		
		state_list = []
		for i in range(self.task.env.height):
			for j in range(self.task.env.width):
				state_list.append(self.task.env.to_obs((i, j), self.sample_path['goal']))
		mu, _ = self.network(np.array(state_list), 'mu')
		option_features_maps = mu.view(self.task.env.height, self.task.env.width, -1).data.numpy()
		
		for option_feature in range(self.config.option_feature_dim):
			ax = axs[2, option_feature]
			im = ax.imshow(option_features_maps[:, :, option_feature], cmap='Blues')
			ax.axis('off')
			fig.colorbar(im, ax=ax)
		# im.set_clim(0, 1)

		plt.savefig("termination_%d.pdf" % self.config.sweep_id)
		# plt.show(block=False)
		# plt.pause(1)
		plt.close()
	
	def get_l1_loss(self):
		# reg_l1_loss = 0
		# for param in self.network.parameters():
		# 	param_loss = torch.sum(torch.abs(param))
		# 	reg_l1_loss += param_loss
		# 	self.total_reg_l1_loss += param_loss.data.numpy()
		# return reg_l1_loss
		
		reg_l1_loss = 0
		for param in self.network.pi_parameters:
			param_loss = torch.sum(torch.abs(param))
			reg_l1_loss += param_loss
			self.total_reg_l1_loss += param_loss.data.numpy()
		return reg_l1_loss
	
	# reg_l1_loss = 0
	# for param in list(self.network.mu_body.parameters()) + list(self.network.fc_mu.parameters()) + \
	#              list(self.network.pi_body.parameters()) + list(self.network.beta_body.parameters()):
	# 	param_loss = torch.sum(torch.abs(param))
	# 	reg_l1_loss += param_loss
	# 	self.total_reg_l1_loss += param_loss.data.numpy()
	# return reg_l1_loss
	
	def get_orthogonal_loss(self):
		reg_orthogonal_loss = 0
		batch_size = 32
		if self.config.reg_orthogonal_weight != 0 and self.replay.size() > 2 * batch_size:
			random_states = self.replay.sample(2 * batch_size)[0]
			option_features, _ = self.network(self.config.state_normalizer(random_states), 'mu')
			a = option_features[:batch_size]
			b = option_features[batch_size:]
			reg_orthogonal_loss += torch.bmm(a.view(batch_size, 1, self.config.option_feature_dim),
			                                 b.view(batch_size, self.config.option_feature_dim, 1)).pow(2).view(batch_size) - \
			                       torch.sum(a ** 2, 1) - \
			                       torch.sum(b ** 2, 1) + \
			                       self.config.option_feature_dim
			reg_orthogonal_loss = reg_orthogonal_loss.mean()
			
			self.total_reg_orthogonal_loss += reg_orthogonal_loss.data.numpy()
		return reg_orthogonal_loss
	
	def get_decorrelated_loss(self):
		reg_decorrelated_loss = 0
		batch_size = 64
		if self.config.reg_decorrelated_weight != 0 and self.replay.size() > batch_size * 6:
			random_states = self.replay.sample(6 * batch_size)[0]
			option_features = self.network(np.expand_dims(self.config.state_normalizer(random_states), axis=0), 'mu')[0]
			a = option_features[:batch_size]
			b = option_features[batch_size:2 * batch_size]
			c = option_features[2 * batch_size:3 * batch_size]
			d = option_features[3 * batch_size:4 * batch_size]
			e = option_features[4 * batch_size:5 * batch_size]
			f = option_features[5 * batch_size:6 * batch_size]
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
		reg_mine_loss = 0
		if self.config.reg_mine_weight != 0:
			# random_states = self.replay.sample(32)[0]
			# option_features = self.network(self.config.state_normalizer(random_states), 'mu')
			# pi, log_pi = self.network((self.config.state_normalizer(random_states), option_features), 'pi')
			# random_states2 = self.replay.sample(32)[0]
			# option_features2 = self.network(self.config.state_normalizer(random_states2), 'mu')
			# # option_features2 = option_features[torch.randperm(32)]
			# pred_xy = self.network((pi, option_features), 'mine')
			# pred_x_y = self.network((pi, option_features2), 'mine')
			# reg_mine_loss = - (torch.mean(pred_xy) - torch.logsumexp(pred_x_y, dim=0))
			# self.total_reg_mine_loss += reg_mine_loss.data.numpy()
			
			random_states = self.replay.sample(64)[0]
			option_features = self.network(self.config.state_normalizer(random_states), 'mu')
			random_states2 = self.replay.sample(64)[0]
			# option_features2 = option_features[torch.randperm(32)]
			pred_xy = self.network((random_states, option_features), 'mine')
			pred_x_y = self.network((random_states2, option_features), 'mine')
			reg_mine_loss = - (torch.mean(pred_xy) - torch.logsumexp(pred_x_y, dim=0))
			self.total_reg_mine_loss += reg_mine_loss.data.numpy()
		return reg_mine_loss