#######################################################################
# Copyright (C) 2020 Yi Wan(wan6@ualberta.ca)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .BaseAgent import *
import matplotlib.pyplot as plt
from ..component.replay import Storage


class ValueComplexityOptionCritic(BaseAgent):
	def __init__(self, config):
		BaseAgent.__init__(self, config)
		self.config = config
		self.task = config.task_fn(config.seed)
		self.network = config.network
		# self.opt = config.optimizer_fn(self.network.parameters())
		self.option_opt = torch.optim.Adam(self.network.option_parameters, lr=config.option_learning_rate,
		                                   eps=config.optimizer_eps)
		self.policy_opt = torch.optim.Adam(self.network.policy_parameters, lr=config.policy_learning_rate,
		                                   eps=config.optimizer_eps)
		self.critic_opt = torch.optim.Adam(self.network.critic_parameters, lr=config.critic_learning_rate,
		                                   eps=config.optimizer_eps)
		
		# self.option_opt = torch.optim.SGD(self.network.option_parameters, lr=config.option_learning_rate)
		# self.policy_opt = torch.optim.SGD(self.network.policy_parameters, lr=config.policy_learning_rate)
		# self.critic_opt = torch.optim.SGD(self.network.critic_parameters, lr=config.critic_learning_rate)
		
		self.total_steps = 0
		self.roll_out_steps = 0
		self.storage = Storage(config.rollout_length)
		self.eps_steps = None
		self.gamma_power = None

		self.prev_option_test = None
		self.state = self.task.reset()
		self.state = tensor([self.config.state_normalizer(self.state)])
		mu_dist, mu_cost, mu_ent = self.network(self.state, 'mu')
		self.option = mu_dist.sample().unsqueeze(-1)
		self.eps_computation_cost = to_np(mu_cost).item()
		self.avg_episodic_length = 0
		
		self.avg_option_prob = torch.ones_like(mu_dist.probs) / self.config.num_options
		self.avg_option_selection_weighting = torch.ones_like(mu_dist.probs) / self.config.num_options
		self.avg_option_cost = torch.zeros_like(mu_dist.probs)
		
		self.eval_cum_cost = 0
		
		self.num_imgs_to_draw = 4
		self.draw_img_idx = 1
		self.to_draw = None
		self.eps_option_list = []

		self.start()

	def step(self):
		self.total_steps += 1
		self.eps_steps += 1
		self.roll_out_steps += 1

		eps_term, option_term, report = self.interact()

		if self.roll_out_steps >= self.config.rollout_length:
			self.train()
			self.roll_out_steps = 0
			self.storage = Storage(self.config.rollout_length)

		return report

	def interact(self):
		report = {}
		self.total_steps += 1
		self.eps_steps += 1

		# Obtain pi(S_t, O_t) and pi_cost(S_t, O_t)
		pi_dist, pi_cost, pi_ent = self.network((self.state, to_np(self.option).item()), 'pi')
		action = pi_dist.sample().unsqueeze(0)
		self.eps_computation_cost += to_np(pi_cost).item()

		# Obtain S_{t+1}, R_{t+1}, and whether or not the episode terminates
		next_state, reward, eps_term, info = self.task.step(to_np(action).item())
		next_state = tensor([self.config.state_normalizer(next_state)])
		reward = tensor([[reward]])

		# obtain beta(S_{t+1}, O_t) and beta_cost(S_{t+1}, O_t)
		if self.config.num_options == 1:
			# The degenerate case, no termination
			next_beta_dist = torch.distributions.Categorical(probs=tensor([[1.0, 0.0]]))
			next_beta_cost = tensor([[0.0]])
			next_beta_ent = tensor([[0.0]])
		else:
			# The normal case when the number of options is greater than 1
			next_beta_dist, next_beta_cost, next_beta_ent = self.network(
				(next_state, to_np(self.option).item()), 'beta'
			)
		
		if eps_term:
			option_term = tensor([[1]])
			next_beta = tensor([[1.0]])
			next_beta_cost = tensor([[0.0]])
		else:
			option_term = next_beta_dist.sample().unsqueeze(-1)
			next_beta = next_beta_dist.probs[:, 1].unsqueeze(-1)
			self.eps_computation_cost += to_np(next_beta_cost).item()

		# Obtain mu(S_{t+1}) and mu_cost(S_{t+1})
		if self.config.num_options == 1:
			# The degenerate case, with only one option and no re-selection of option is needed
			next_mu_dist = torch.distributions.Categorical(probs=tensor([[1.0]]))
			next_mu_cost = tensor([[0.0]])
			next_mu_ent = tensor([[0.0]])
		else:
			next_mu_dist, next_mu_cost, next_mu_ent = self.network(next_state, 'mu')
		
		# enforce using under-used options
		if self.config.enforce_under_used_options:
			tmp = torch.zeros_like(self.avg_option_prob)
			tmp[0, self.option] = 1
			self.avg_option_prob = 0.999 * self.avg_option_prob + 0.001 * tmp
			if torch.min(self.avg_option_prob) < 0.05:
				next_mu_ent = 10 * next_mu_dist.entropy().unsqueeze(-1)
				under_used_options = self.avg_option_prob < 0.05
				next_mu_dist = torch.distributions.Categorical(
					probs=under_used_options.float() / under_used_options.sum().float()
				)
				option_term = torch.tensor([[1]])
				next_beta = tensor([[1.0]])
				next_beta_ent = 10 * next_beta_dist.entropy().unsqueeze(-1)
			
		if option_term:
			next_option = next_mu_dist.sample().unsqueeze(-1)
			self.eps_computation_cost += to_np(next_mu_cost).item()
		else:
			next_option = self.option

		# Store experience for n-step learning
		self.storage.add({
			'state': self.state,
			'action': action,
			'log_pi': pi_dist.log_prob(action),
			'pi_cost': pi_cost,
			'pi_ent': pi_ent,
			'next_beta': next_beta,
			'next_beta_cost': next_beta_cost,
			'next_beta_ent': next_beta_ent,
			'next_mu': next_mu_dist.probs,
			'next_mu_cost': next_mu_cost,
			'next_mu_ent': next_mu_ent,
			'reward': reward,
			'option': self.option,
			'eps_term': tensor([[eps_term]]),
			'gamma_power': self.gamma_power,
		})
		
		self.state = next_state
		self.option = next_option

		if eps_term:
			self.avg_episodic_length = self.avg_episodic_length * 0.99 + info['episodic_length'] * 0.01
			report.setdefault('episodic_cumulative_reward', []).append(info['episodic_return'])
			report.setdefault('episodic_time_complexity', []).append(self.eps_computation_cost)
			report.setdefault('space_complexity', []).append(to_np(self.get_space_complexity()).item())
			report.setdefault('episodic_length', []).append(info['episodic_length'])
			self.start()

		return eps_term, option_term, report

	def train(self):
		gamma = self.config.discount
		eta = self.config.tc_weight
		
		# Extract experience for training
		action, reward, state, option, next_beta, log_pi, gamma_power, next_mu, eps_term, \
		pi_cost, next_mu_cost, next_beta_cost, pi_ent, next_mu_ent, next_beta_ent = self.storage.cat(
			[
				'action', 'reward', 'state', 'option', 'next_beta', 'log_pi', 'gamma_power',
				'next_mu', 'eps_term', 'pi_cost', 'next_mu_cost', 'next_beta_cost',
				'pi_ent', 'next_mu_ent', 'next_beta_ent'
			]
		)
		
		critic_input = torch.cat((state, self.state), dim=0)
		if self.config.critic_input == 'tabular':
			critic_input = tensor(
				self.config.state_normalizer(
					self.task.env.xy_to_tabular(
						critic_input.numpy()
					)
				)
			)
		q = self.network(critic_input, 'q')
		
		# q(S_t, O_t) q(S_{t+1}, O_{t+1}) ... q(S_{t+H-1}, S_{t+H-1})
		q_curr_s_curr_o = q[:-1].gather(1, option)
		
		if self.config.critic_eval == "off-option":
			"""
			TBD
			"""
			# compute importance sampling ratio
			pi_dist_batch, pi_cost_batch, pi_ent_batch = self.network(state, 'pi')
			pi_a_batch = pi_dist_batch.probs.gather(
				2, action.expand(self.roll_out_steps, self.config.num_options).unsqueeze(-1)
			).squeeze(2)
			rho_batch = pi_a_batch / (pi_a_batch.gather(1, option) + 0.0001)
			rho_batch = torch.clamp(rho_batch, max=10)
			next_beta_dist_batch, next_beta_cost_batch, next_beta_ent_batch = self.network(
				torch.cat((state[1:], self.state), dim=0), 'beta'
			)
			# compute bootstrapping value
			# q_(S_{t+H}, o)
			ret = q[-1].unsqueeze(0)
			rets = []
			for i in reversed(range(self.roll_out_steps)):
				ret = - pi_cost_batch[i] + rho_batch[i] * reward[i] + rho_batch[i] * gamma * (1 - eps_term[i]) * (
					- next_beta_cost_batch[i]
					+ next_beta_dist_batch.probs[i, :, 1] * (-next_mu_cost[i] + torch.dot(next_mu[i], ret[0]))
					+ next_beta_dist_batch.probs[i, :, 0] * ret
				)
				# ret = reward[i] + self.config.discount * (1 - eps_term[i]) * ret
				
				rets.append(ret)
			rets.reverse()
			rets = torch.cat(rets, dim=0)
			
			# off-option critic loss
			q_loss = ((rets.detach() - q[:-1]).pow(2).mul(0.5)).mean()
			
			rets = rets.gather(1, option)
		elif self.config.critic_eval == "on-option":
			"""
			compute bootstrapping value
			Bellman-like Equation:
			q(s, o) = E[- eta * pi_cost(S_t, O_t) + R_{t+1} + gamma * (
					- eta * beta_cost(S_{t+1}, O_t) - eta * beta(S_{t+1}, O_t) * mu_cost(S_{t+1}) + q(S_{t+1}, O_{t+1})
					) | S_t = s, O_t = o]
			"""
			# q_(S_{t+H}, O_{t+H})
			ret = q[-1].unsqueeze(0).gather(1, self.option)
			rets = []
			for i in reversed(range(self.roll_out_steps)):
				ret = - eta * pi_cost[i] + reward[i] + gamma * (1 - eps_term[i]) * (
					- eta * next_beta_cost[i]
					- eta * next_beta[i] * next_mu_cost[i]
					+ ret
				)
				rets.append(ret)
			rets.reverse()
			rets = torch.cat(rets, dim=0)
			
			# on-option critic loss
			q_loss = ((rets.detach() - q_curr_s_curr_o).pow(2).mul(0.5)).mean()
		else:
			raise NotImplementedError

		# option loss
		pi_loss = (gamma_power * (
				eta * pi_cost
				- log_pi * (rets + eta * pi_cost - q_curr_s_curr_o).detach()
				- self.config.pi_entropy_weight * pi_ent
		)).mean()
		
		# q(S_{t+1}, O_t)
		q_next_s_curr_o = q[1:].gather(1, option)
		
		# expected q(S_{t+1}, .)
		expected_q = (next_mu * q[1:].detach()).sum(1).unsqueeze(1)
		
		# v(S_{t+1})
		next_v = - eta * next_mu_cost + expected_q
		beta_loss = (gamma_power * gamma * (
				eta * next_beta_cost
				+ next_beta * (q_next_s_curr_o - next_v).detach()
				- self.config.beta_entropy_weight * next_beta_ent
		)).mean()

		# policy over options loss
		mu_loss = (
				- (1 - eps_term) * gamma_power * gamma * next_beta.detach() * next_v
				- eps_term * next_v
				- self.config.mu_entropy_weight * next_mu_ent
		).mean()
		
		sc_cost = self.get_space_complexity()

		self.network.zero_grad()
		(pi_loss + beta_loss + mu_loss + q_loss + sc_cost).backward()
		self.critic_opt.step()
		self.option_opt.step()
		self.policy_opt.step()
	
	def get_space_complexity(self):
		# obtain space complexity cost
		if self.config.return_cost == 'l1_weight':
			sc_cost = 0
			for parameters in self.network.option_parameters:
				sc_cost += parameters.abs().sum()
			for parameters in self.network.policy_parameters:
				sc_cost += parameters.abs().sum()
			sc_cost = sc_cost * self.config.sc_weight
		elif self.config.return_cost == 'l0':
			sc_cost = 0
			N = self.network.N
			for o in range(self.config.num_options):
				is_first_layer = True
				for layer in self.network.pi_bodies[o].layers:
					if is_first_layer is False:
						sc_cost += - (1. / N) * layer.regularization()
					is_first_layer = False
				sc_cost += - (1. / N) * self.network.fc_pis[o].regularization()
				is_first_layer = True
				for layer in self.network.beta_bodies[o].layers:
					if is_first_layer is False:
						sc_cost += - (1. / N) * layer.regularization()
					is_first_layer = False
				sc_cost += - (1. / self.network.N) * self.network.fc_betas[o].regularization()
			is_first_layer = True
			for layer in self.network.mu_body.layers:
				if is_first_layer is False:
					sc_cost += - (1. / N) * layer.regularization()
				is_first_layer = False
			sc_cost += - (1. / N) * self.network.fc_mu.regularization()
		else:
			raise NotImplementedError
		
		return sc_cost
		
	def start(self):
		self.eps_computation_cost = 0
		self.eps_steps = 0
		self.gamma_power = tensor([[1]])
	
	def eval_step(self, ep, state, reward, terminal, info):
		
		state = tensor([self.config.state_normalizer(state)])
		
		mu_dist, mu_cost, mu_ent = self.network(state, 'mu')
		
		if terminal is True:
			option_test = mu_dist.sample().unsqueeze(-1)
			if ep == 0:
				self.draw_computational_complexities()
				if "FourRooms" in self.config.task_name:
					self.draw_discovered_options_four_rooms()
				elif ("Catcher" in self.config.task_name or "Pixelcopter" in self.config.task_name):
					# print(self.draw_img_idx)
					if self.to_draw is not None:
						if self.draw_img_idx == 1:
							self.to_draw = np.clip(self.exp_avg_img, a_min=0, a_max=255).astype(np.uint8)
						elif self.draw_img_idx <= self.num_imgs_to_draw:
							self.to_draw = np.concatenate(
								(
									self.to_draw,
									np.ones((3, self.to_draw.shape[1], 5), dtype=np.uint8) * 255,
									np.clip(self.exp_avg_img, a_min=0, a_max=255).astype(np.uint8)
								), 2
							)
						
						self.options_to_draw = np.ones((3, self.to_draw.shape[1], 5), dtype=np.uint8) * \
						                       self.eps_option_list[0] * 255
						for i in self.eps_option_list[1:]:
							tmp = np.ones((3, self.to_draw.shape[1], 5), dtype=np.uint8) * i * 255
							self.options_to_draw = np.concatenate((self.options_to_draw, tmp), 2)
						self.to_draw = np.concatenate(
							(self.to_draw, np.zeros((3, 10, self.to_draw.shape[2]), dtype=np.uint8)), 1)
						self.eps_option_list = []
						fig, axs = plt.subplots(2, 1)
						axs[0].imshow(np.transpose(self.to_draw, (1, 2, 0)))
						axs[0].axis('off')
						axs[1].imshow(np.transpose(self.options_to_draw, (1, 2, 0)))
						axs[1].axis('off')
						tag = "Discovered_Options"
						plt.savefig(
							"experiment/%s/outputs/%s_%d.pdf" % (self.config.exp_name, tag, self.config.sweep_id))
						plt.close()
						self.logger.writer.add_image("option_img", self.options_to_draw, self.total_steps)
						self.logger.writer.add_image("img", self.to_draw, self.total_steps)
						self.logger.writer.flush()
					# img = self.config.eval_task.env.gameOb.getScreenRGB()
					# self.exp_avg_img = np.transpose(img, (2, 1, 0))
					self.exp_avg_img = None
					self.draw_img_idx = 1
		else:
			if ("Catcher" in self.config.task_name or "Pixelcopter" in self.config.task_name) and ep == 0:
				img = self.config.eval_task.env.gameOb.getScreenRGB()
				if self.exp_avg_img is None:
					self.exp_avg_img = np.transpose(img, (2, 1, 0))
				else:
					self.exp_avg_img = np.transpose(img, (2, 1, 0)) + 0.7 * self.exp_avg_img
			
			beta_dist, beta_cost, beta_ent = self.network((state, to_np(self.prev_option_test).item()), 'beta')
			prev_option_term = beta_dist.sample()
			
			if prev_option_term:
				option_test = mu_dist.sample().unsqueeze(-1)
				if option_test != self.prev_option_test and (
						"Catcher" in self.config.task_name or "Pixelcopter" in self.config.task_name) and ep == 0:
					
					if self.draw_img_idx == 1:
						self.to_draw = np.clip(self.exp_avg_img, a_min=0, a_max=255).astype(np.uint8)
					elif self.draw_img_idx <= self.num_imgs_to_draw:
						self.to_draw = np.concatenate(
							(
								self.to_draw,
								np.ones((3, self.to_draw.shape[1], 5), dtype=np.uint8) * 255,
								np.clip(self.exp_avg_img, a_min=0, a_max=255).astype(np.uint8)
							), 2
						)
					img = self.config.eval_task.env.gameOb.getScreenRGB()
					self.exp_avg_img = np.transpose(img, (2, 1, 0))
					self.draw_img_idx += 1
			
			else:
				option_test = self.prev_option_test
		
		pi_dist, pi_cost, pi_ent = self.network((state, to_np(option_test).item()), 'pi')
		action = pi_dist.sample()
		
		self.prev_option_test = option_test
		
		self.eps_option_list.append(self.prev_option_test.data.numpy().item())
		
		return to_np(action).item()

	def draw_discovered_options_four_rooms(self):
		q_maps = [self.task.env.occupancy.astype('float64') for _ in range(self.config.num_options)]
		adv_maps = [self.task.env.occupancy.astype('float64') for _ in range(self.config.num_options)]
		beta_maps = [self.task.env.occupancy.astype('float64') for _ in range(self.config.num_options)]
		beta_cost_maps = [self.task.env.occupancy.astype('float64') * 0 for _ in range(self.config.num_options)]
		mu_maps = [self.task.env.occupancy.astype('float64') for _ in range(self.config.num_options)]
		pi_y = [
			[self.task.env.occupancy.astype('float64') for _ in range(self.task.env.action_space.n)]
			for _ in range(self.config.num_options)
		]
		pi_x = [
			[self.task.env.occupancy.astype('float64') for _ in range(self.task.env.action_space.n)]
			for _ in range(self.config.num_options)
		]
		pi_cost_maps = [self.task.env.occupancy.astype('float64') * 0 for _ in range(self.config.num_options)]
		mu_cost_maps = [self.task.env.occupancy.astype('float64') * 0 for _ in range(self.config.num_options)]

		state_list = []
		for i in range(self.task.env.height):
			for j in range(self.task.env.width):
				state_list.append(self.task.env.to_obs((i, j), self.task.env.tocell[self.task.env.goals[0]]))
		states = tensor(state_list)
		mu_dist, mu_cost, mu_ent = self.network(states, 'mu')
		
		critic_input = states
		if self.config.critic_input == 'tabular':
			critic_input = tensor(
				self.config.state_normalizer(
					self.task.env.xy_to_tabular(
						critic_input.numpy()
					)
				)
			)
		q = self.network(critic_input, 'q')
		v = (mu_dist.probs * q.detach()).sum(1).unsqueeze(1)
		pi_dist, pi_cost, pi_ent = self.network(states, 'pi')
		beta_dist, beta_cost, beta_ent = self.network(states, 'beta')
		
		avg_mu = [0 for _ in range(self.config.num_options)]
		num_states = 0
		for option in range(self.config.num_options):
			for i in range(self.task.env.height):
				for j in range(self.task.env.width):
					if beta_maps[option][i, j] == 0:
						q_maps[option][i, j] = to_np(q[i * self.task.env.width + j, option])
						adv_maps[option][i, j] = to_np(
							q[i * self.task.env.width + j, option] - v[i * self.task.env.width + j]
						)
						beta_maps[option][i, j] = to_np(beta_dist.probs[i * self.task.env.width + j, option, 1])
						mu_maps[option][i, j] = to_np(mu_dist.probs[i * self.task.env.width + j, option])
						mu_cost_maps[option][i, j] = to_np(mu_cost[i * self.task.env.width + j])
						avg_mu[option] = avg_mu[option] + mu_maps[option][i, j]
						num_states += 1
						beta_cost_maps[option][i, j] = to_np(beta_cost[i * self.task.env.width + j, option])
						pi_cost_maps[option][i, j] = to_np(pi_cost[i * self.task.env.width + j, option])
						for a in range(self.task.env.action_space.n):
							y, x = self.task.env.directions[a]
							magnitude = to_np(pi_dist.probs[i * self.task.env.width + j, option, a])
							pi_y[option][a][i, j] = -y * magnitude
							pi_x[option][a][i, j] = x * magnitude

					else:
						for a in range(self.task.env.action_space.n):
							pi_y[option][a][i, j] = 0
							pi_x[option][a][i, j] = 0
		
		rows = ['values', 'adv', 'pis', 'betas', 'mu']
		fig, axs = plt.subplots(len(rows), self.config.num_options, figsize=(10, 10))
		# fig.suptitle("avg_episodic_length %f" % self.avg_episodic_length, fontsize=16)
		
		# option value
		min_q = np.floor(np.min([np.min(q_maps[o]) for o in range(self.config.num_options)]))
		max_q = np.ceil(np.max([np.max(q_maps[o]) for o in range(self.config.num_options)]))
		for o in range(self.config.num_options):
			if self.config.num_options == 1:
				ax = axs[rows.index('values')]
			else:
				ax = axs[rows.index('values'), o]
			im = ax.imshow(q_maps[o], cmap='Blues')
			ax.axis("off")
			im.set_clim(min_q, max_q)
		fig.colorbar(im, ax=ax)
		
		# advantage
		for o in range(self.config.num_options):
			if self.config.num_options == 1:
				ax = axs[rows.index('adv')]
			else:
				ax = axs[rows.index('adv'), o]
			im = ax.imshow(adv_maps[o], cmap='bwr')
			ax.axis("off")
			im.set_clim(-1, 1)
		fig.colorbar(im, ax=ax)
		
		# pi
		Y = np.arange(0, self.task.env.height, 1)
		X = np.arange(0, self.task.env.width, 1)
		cell_map = self.task.env.occupancy.astype('float64')
		goal_cell = self.task.env.tocell[self.task.env.goals[0]]
		cell_map[goal_cell] = 1
		for o in range(self.config.num_options):
			if self.config.num_options == 1:
				ax = axs[rows.index('pis')]
			else:
				ax = axs[rows.index('pis'), o]
			im = ax.imshow(cell_map, cmap='Blues')
			for a in range(self.task.env.action_space.n):
				ax.quiver(X, Y, pi_x[o][a], pi_y[o][a], scale=15.0)
			ax.axis('off')
		fig.colorbar(im, ax=ax)

		# beta
		for o in range(self.config.num_options):
			if self.config.num_options == 1:
				ax = axs[rows.index('betas')]
			else:
				ax = axs[rows.index('betas'), o]
			im = ax.imshow(beta_maps[o], cmap='Blues')
			ax.axis('off')
			
			im.set_clim(0, 1)
		fig.colorbar(im, ax=ax)
		
		# mu
		for o in range(self.config.num_options):
			if self.config.num_options == 1:
				ax = axs[rows.index('mu')]
			else:
				ax = axs[rows.index('mu'), o]
			im = ax.imshow(mu_maps[o], cmap='Blues')
			ax.axis('off')
			im.set_clim(0, 1)
		fig.colorbar(im, ax=ax)
		
		tag = "Discovered_Options"
		plt.savefig("experiment/%s/outputs/%s_%d.pdf" %(self.config.exp_name, tag, self.config.sweep_id))
		plt.close()
		
		self.logger.writer.add_figure(tag=tag, figure=fig, global_step=self.total_steps)
		self.logger.writer.flush()
	
	def draw_computational_complexities(self):
		pi_norm_list = []
		beta_norm_list = []
		if self.config.return_cost == 'l1_weight':
			mu_norm = 0
			for params in self.network.mu_params:
				mu_norm += params.abs().sum()
			mu_norm_list = mu_norm.data.numpy().flatten().tolist()
			
			for option in range(self.config.num_options):
				pi_norm = 0
				for params in self.network.pi_params[option]:
					pi_norm += params.abs().sum()
				pi_norm_list += pi_norm.data.numpy().flatten().tolist()
				
				beta_norm = 0
				for params in self.network.beta_params[option]:
					beta_norm += params.abs().sum()
				beta_norm_list += beta_norm.data.numpy().flatten().tolist()
			label = 'l1 norm'
		elif self.config.return_cost == 'l0':
			mu_norm = 0
			for params in self.network.mu_params:
				mu_norm += (params.abs() > 0.001).sum()
			mu_norm_list = mu_norm.data.numpy().flatten().tolist()
	
			for option in range(self.config.num_options):
				pi_norm = 0
				for params in self.network.pi_params[option]:
					pi_norm += (params.abs() > 0.001).sum()
				pi_norm_list += pi_norm.data.numpy().flatten().tolist()
				
				beta_norm = 0
				for params in self.network.beta_params[option]:
					beta_norm += (params.abs() > 0.001).sum()
				beta_norm_list += beta_norm.data.numpy().flatten().tolist()
			label = 'l0 norm'
		else:
			raise NotImplementedError
		
		width = 0.35
		
		fig, ax = plt.subplots()
		x = np.arange(0, self.config.num_options)
		ax.bar(x - width/2, pi_norm_list, width=0.2, label=r'$\pi$')
		ax.bar(x + width/2, beta_norm_list, width=0.2, label=r'$\beta$')
		ax.bar(np.arange(self.config.num_options, self.config.num_options + 1), mu_norm_list, width=0.2, label=r'$\mu$')
		ax.set_ylabel(label)
		ax.set_xticks(x)
		labels = ['option %d' %i for i in range(self.config.num_options)]
		ax.set_xticklabels(labels)
		# ax.set_title(r'computational complexity')
		ax.legend()

		fig.tight_layout()
		
		tag = "Computational_Complexities"
		
		plt.savefig("experiment/%s/outputs/%s_%d.pdf" % (self.config.exp_name, tag, self.config.sweep_id))
		plt.close()
		
		self.logger.writer.add_figure(tag=tag, figure=fig, global_step=self.total_steps)
		self.logger.writer.flush()