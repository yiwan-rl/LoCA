#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl.agent.BaseAgent import *
import matplotlib.pyplot as plt
import torch
from deep_rl.component.replay import Storage


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
		self.state = tensor([self.config.state_normalizer(self.state)])
		
		self.roll_out_steps = 0
		self.storage = Storage(100)
		
		self.switch_cost = 0
		
		if config.reg_orthogonal_weight != 0 or config.reg_decorrelated_weight != 0 or config.reg_mine_weight != 0:
			self.replay = config.replay_fn()
			
		# DEBUG
		self.total_eps = 0
		
		self.start()
	
	def step(self):
		config = self.config
		report = {}
		
		self.sample_path['cell'].append(self.task.env.current_cell)
		self.sample_path['option'].append(self.option.detach())
		
		self.total_steps += 1
		self.eps_steps += 1
		self.roll_out_steps += 1
		
		pi_dist = self.network((self.state, self.option.detach()), 'pi')
		action = pi_dist.sample()[0]
		log_pi = pi_dist.log_prob(action).unsqueeze(0)
		pi_entropy = pi_dist.entropy().unsqueeze(0)
		
		next_state, reward, terminal, info = self.task.step(to_np(action).item())
		
		next_state = tensor([self.config.state_normalizer(next_state)])
		reward = tensor([[reward - config.switch_cost]])
		
		# terminate option
		beta_dist = self.network((next_state, self.option.detach()), 'beta')
		next_beta = beta_dist.probs
		beta_entropy = beta_dist.entropy()
		option_term = beta_dist.sample()[0, 0]
		
		self.storage.add({
			'next_beta': next_beta, 'beta_entropy': beta_entropy,
			'log_pi': log_pi, 'pi_entropy': pi_entropy, 'mu_entropy': self.mu_entropy,
			'reward': reward, 'option': self.option, 'state': self.state, 'gamma_power': self.gamma_power
		})
		
		if not terminal:
			
			if option_term:
				next_option, _ = self.network(next_state, 'mu')
				self.switch_cost = config.switch_cost
			else:
				next_option = self.option
				self.switch_cost = 0
			
			if config.task_name == "TwoRoomsThreeGoals":
				self.sample_path['option_term'].append(option_term)
		else:
			next_option = None
			if config.task_name == "TwoRoomsThreeGoals":
				self.sample_path['option_term'].append(1)
		
		self.state = next_state
		self.option = next_option
		self.gamma_power = self.gamma_power * config.discount
		
		if terminal or (self.roll_out_steps >= config.rollout_length and option_term):
			
			states, options = self.storage.cat(['state', 'option'])
			next_states = torch.cat((states[1:], self.state), dim=0)
			
			mu_next_states, mu_entropy = self.network(next_states, 'mu')
	
			q = self.network((torch.cat((states, next_states, next_states), dim=0), torch.cat((options, options, mu_next_states), dim=0)), 'q')
			q_s_o = q[:self.roll_out_steps]
			q_next_s_o = q[self.roll_out_steps:self.roll_out_steps*2]
			v_next_s = q[self.roll_out_steps*2:]
			
			# compute bootstrapping value
			if terminal:
				ret = 0
			else:
				ret = v_next_s[-1]
			
			rets = []
			for i in reversed(range(self.roll_out_steps)):
				ret = self.storage.reward[i] + config.discount * ret
				rets.append(ret)
			rets.reverse()
			rets = torch.cat(rets, dim=0)
			next_beta, beta_entropy, log_pi, pi_entropy, gamma_power = self.storage.cat(['next_beta', 'beta_entropy', 'log_pi', 'pi_entropy', 'gamma_power'])
			
			q_loss = (rets.detach() - q_s_o).pow(2).mul(0.5).mean()
			
			# optimize q
			self.network.zero_grad()
			(
					config.q_weight * q_loss
			).backward(retain_graph=True)
			self.q_opt.step()
			
			# optimize pi, beta, mu
			pi_loss = - gamma_power * log_pi * (rets - q_s_o).detach() - gamma_power * config.pi_entropy_weight * pi_entropy
			pi_loss = pi_loss.mean()
			beta_loss = config.discount * gamma_power * next_beta * (q_next_s_o - v_next_s + config.switch_cost).detach() - gamma_power * config.beta_entropy_weight * beta_entropy
			mu_loss = - config.discount * gamma_power * next_beta.detach() * v_next_s - config.mu_entropy_weight * gamma_power * mu_entropy
			
			if terminal:
				if beta_loss.shape[0] > 1:
					beta_loss = beta_loss[:-1].mean()
					mu_loss = mu_loss[:-1].mean()
				else:
					beta_loss = 0
					mu_loss = 0
			else:
				beta_loss = beta_loss.mean()
				mu_loss = mu_loss.mean()
			
			# optimize
			self.network.zero_grad()
			(
				config.pi_beta_weight * (pi_loss + beta_loss) +
				config.mu_weight * mu_loss
			).backward()
			
			self.pi_opt.step()
			self.beta_opt.step()
			self.mu_opt.step()
			
			self.roll_out_steps = 0
			self.storage = Storage(100)
		
		if terminal:
			# record info
			report.setdefault('episodic_return', []).append(info['episodic_return'])
			report.setdefault('episodic_length', []).append(info['episodic_length'])
			if config.task_name == "TwoRoomsThreeGoals" and self.total_eps % config.print_eps_interval == 0 and config.rank == 0:
				self.print_sample_path()
			self.start()
		
		return report
	
	def start(self):
		self.gamma_power = tensor([[1]])
		self.total_eps += 1
		self.eps_steps = 0
		self.switch_cost = 0
		
		option, mu_entropy = self.network(self.state, 'mu')
		v_next_s = self.network((self.state, option), 'q')
		mu_loss = - v_next_s - self.config.mu_entropy_weight * mu_entropy
		
		# optimize
		self.network.zero_grad()
		(
				self.config.mu_weight * mu_loss.mean()
		).backward()
		
		self.mu_opt.step()
		
		self.option, self.mu_entropy = self.network(self.state, 'mu')
		
		# DEBUG and PRINT
		self.sample_path = {
			'cell': [], 'option': [], 'option_term': [],
			'goal': self.task.env.tocell[self.task.env.goals[self.task.env.goal_idx]]
		}
		
	def print_sample_path(self):
		num_terms_to_print = self.config.option_feature_dim
		curr_term = 0
		cell_map = self.task.env.occupancy.astype('float64')
		cell_map[self.sample_path['goal'][0], self.sample_path['goal'][1]] = 1
		cell_map[self.sample_path['cell'][0][0], self.sample_path['cell'][0][1]] = 0.3
		
		fig, axs = plt.subplots(3, num_terms_to_print, figsize=(20, 10))
		
		for idx in range(len(self.sample_path['option_term'])):
			if self.sample_path['option_term'][idx] == 0:
				cell_map[self.sample_path['cell'][idx+1][0], self.sample_path['cell'][idx+1][1]] = 0.3
				continue
				
			# sample path
			if idx != len(self.sample_path['option_term']) - 1:
				cell_map[self.sample_path['cell'][idx+1][0], self.sample_path['cell'][idx+1][1]] = 0.6
			
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