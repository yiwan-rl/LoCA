#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
from deep_rl.agent.muzero_helpers import *
from torch.nn import L1Loss
import torch.optim as optim
import ray


train_logger = logging.getLogger('train')
evaluate_logger = logging.getLogger('train_evaluate')


class MuZero(BaseAgent):
	def __init__(self, config, muzero_config, domain_settings, summary_writer):
		BaseAgent.__init__(self, config)
		self.config = config
		self.muzero_config = muzero_config
		self.summary_writer = summary_writer
		# self.task = config.task_fn(config.seed)
		self.network = config.network
		self.optimizer = optim.SGD(
			self.network.parameters(),
			lr=muzero_config.lr_init,
			momentum=muzero_config.momentum,
			weight_decay=muzero_config.weight_decay
		)
		# self.optimizer = set_optimizer(self.network.parameters(), config)
		self.step_count = 0
		# self.states = self.task.reset()
		ray.init()
		self.shared_storage = SharedStorage.remote(self.network)
		self.replay_buffer = ReplayBuffer.remote(
			batch_size=muzero_config.batch_size,
			capacity=muzero_config.window_size,
			prob_alpha=muzero_config.priority_prob_alpha
		)
		self.workers = []
		for rank in range(0, muzero_config.num_actors):
			worker = DataWorker.remote(
				rank, muzero_config, domain_settings, self.shared_storage, self.replay_buffer
			).run.remote()
			self.workers.append(worker)

		self.workers += [_evaluate.remote(muzero_config, domain_settings, self.shared_storage)]

		self.target_network = muzero_config.get_uniform_network().to('cpu')
		self.target_network.eval()
		#
		
	
		#
		# model = config.get_uniform_network().to(config.device)
		# model.train()
		# optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
		#                       weight_decay=config.weight_decay)
		# self.target_network = muzero_config.get_uniform_network().to('cpu')
		# self.target_network.eval()
		#
		# # wait for replay buffer to be non-empty
		# while ray.get(self.replay_buffer.size.remote()) == 0:
		# 	pass
		#
		# for step_count in range(config.training_steps):
	
	def eval_step(self, ep, state, reward, done, info):
		if done:
			state = self.config.eval_env.env.reset_test()
		prediction = self.network(np.expand_dims(self.config.state_normalizer(state), axis=0))
		return to_np(prediction['a']).item()
	
	def step(self):
		# report = {'episodic_return': [], 'actor_loss': [], 'critic_loss': [], 'max_a_prob': []}
		# config = self.config
		# storage = Storage(config.rollout_length)
		# for _ in range(config.rollout_length):
		# 	prediction = self.network(np.expand_dims(config.state_normalizer(self.states), axis=0))
		# 	next_states, reward, terminal, info = self.task.step(to_np(prediction['a']).item())
		# 	# self.record_online_return(info)
		# 	ret = info['episodic_return']
		# 	if ret is not None:
		# 		report['episodic_return'].append(ret)
		# 		report.setdefault('episodic_length', []).append(info['episodic_length'])
		# 	report['max_a_prob'].append(to_np(prediction['max_a_prob']).item())
		# 	reward = config.reward_normalizer(reward)
		# 	storage.add(prediction)
		# 	storage.add({'r': tensor(reward), 'm': tensor(1 - terminal)})
		# 	self.states = next_states
		# 	self.total_steps += 1
		#
		# prediction = self.network(np.expand_dims(config.state_normalizer(self.states), axis=0))
		# storage.add(prediction)
		# storage.placeholder()
		#
		# advantages = tensor(np.zeros((1, 1)))
		# returns = prediction['v'].detach()
		# for i in reversed(range(config.rollout_length)):
		# 	returns = storage.r[i] + config.discount * storage.m[i] * returns
		# 	if not config.use_gae:
		# 		advantages = returns - storage.v[i].detach()
		# 	else:
		# 		td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
		# 		advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
		# 	storage.adv[i] = advantages.detach()
		# 	storage.ret[i] = returns.detach()
		#
		# log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
		# policy_loss = -(log_prob * advantages).mean()
		# value_loss = 0.5 * (returns - value).pow(2).mean()
		# entropy_loss = entropy.mean()
		#
		# report['actor_loss'].append(to_np(policy_loss))
		# report['critic_loss'].append(to_np(value_loss))
		#
		# self.optimizer.zero_grad()
		# (policy_loss - config.entropy_weight * entropy_loss + config.value_weight * value_loss).backward()
		# if self.config.gradient_clip is not None:
		# 	nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
		# self.optimizer.step()
		# return report
		
		# wait for replay buffer to be non-empty
		while ray.get(self.replay_buffer.size.remote()) == 0:
			pass
		
		muzero_config = self.muzero_config
		self.shared_storage.incr_counter.remote()
		lr = adjust_lr(muzero_config, self.optimizer, self.step_count)
		
		if self.step_count % self.muzero_config.checkpoint_interval == 0:
			self.shared_storage.set_weights.remote(self.network.get_weights())
		
		log_data = update_weights(self.network, self.target_network, self.optimizer, self.replay_buffer, muzero_config)
		
		# softly update target model
		if muzero_config.use_target_model:
			soft_update(self.target_network, self.network, tau=1e-2)
			self.target_network.eval()
		
		if ray.get(self.shared_storage.get_counter.remote()) % 50 == 0:
			_log(muzero_config, self.step_count, log_data, model, self.replay_buffer, lr,
			     ray.get(self.shared_storage.get_worker_logs.remote()), self.summary_writer)
			
			self.replay_buffer.remove_to_fit.remote()
		performance, _ = ray.get(self.shared_storage.get_evaluate_log.remote())
		
		if self.step_count > 5000 and sum(performance[-100:]) / 100 > 0.98:
			self.shared_storage.step_counter = self.muzero_config.training_steps
			return True
		
		self.step_count += 1
		
		return False
		
		# self.shared_storage.set_weights.remote(model.get_weights())
	

def update_weights(model, target_model, optimizer, replay_buffer, config):
	batch = ray.get(replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps,
	                                                  model=target_model if config.use_target_model else None,
	                                                  config=config))
	obs_batch, action_batch, target_reward, target_value, target_policy, indices, weights = batch
	
	obs_batch = obs_batch.to(config.device)
	action_batch = action_batch.to(config.device).unsqueeze(-1)
	target_reward = target_reward.to(config.device)
	target_value = target_value.to(config.device)
	target_policy = target_policy.to(config.device)
	weights = weights.to(config.device)
	
	# transform targets to categorical representation
	# Reference:  Appendix F
	transformed_target_reward = config.scalar_transform(target_reward)
	target_reward_phi = config.reward_phi(transformed_target_reward)
	transformed_target_value = config.scalar_transform(target_value)
	target_value_phi = config.value_phi(transformed_target_value)
	value, _, policy_logits, hidden_state = model.initial_inference(obs_batch)
	
	scaled_value = config.inverse_value_transform(value)
	# Note: Following line is just for logging.
	predicted_values, predicted_rewards, predicted_policies = scaled_value, None, torch.softmax(policy_logits, dim=1)
	
	# Reference: Appendix G
	new_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
	new_priority += 1e-5
	new_priority = new_priority.data.cpu().numpy()
	
	value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
	policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
	reward_loss = torch.zeros(config.batch_size, device=config.device)
	
	gradient_scale = 1 / config.num_unroll_steps
	for step_i in range(config.num_unroll_steps):
		value, reward, policy_logits, hidden_state = model.recurrent_inference(hidden_state, action_batch[:, step_i])
		policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
		value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
		reward_loss += config.scalar_reward_loss(reward, target_reward_phi[:, step_i])
		hidden_state.register_hook(lambda grad: grad * 0.5)
		
		# collected for logging
		predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value)))
		scaled_rewards = config.inverse_reward_transform(reward)
		predicted_rewards = scaled_rewards if predicted_rewards is None else torch.cat((predicted_rewards,
		                                                                                scaled_rewards))
		predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1)))
	
	# optimize
	loss = (policy_loss + config.value_loss_coeff * value_loss + reward_loss)
	weighted_loss = (weights * loss).mean()
	weighted_loss.register_hook(lambda grad: grad * gradient_scale)
	loss = loss.mean()
	
	optimizer.zero_grad()
	weighted_loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
	optimizer.step()
	
	# update priorities
	replay_buffer.update_priorities.remote(indices, new_priority)
	
	# packing data for logging
	loss_data = (weighted_loss.item(), loss.item(), policy_loss.mean().item(), reward_loss.mean().item(),
	             value_loss.mean().item())
	td_data = (target_reward, target_value, transformed_target_reward, transformed_target_value,
	           target_reward_phi, target_value_phi, predicted_rewards, predicted_values,
	           target_policy, predicted_policies)
	priority_data = (weights, indices)
	
	return loss_data, td_data, priority_data


def adjust_lr(config, optimizer, step_count):
	lr = config.lr_init * config.lr_decay_rate ** (step_count / config.lr_decay_steps)
	lr = max(lr, 0.001)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
		

def _log(config, step_count, log_data, model, replay_buffer, lr, worker_logs, summary_writer):
	loss_data, td_data, priority_data = log_data
	weighted_loss, loss, policy_loss, reward_loss, value_loss = loss_data
	target_reward, target_value, trans_target_reward, trans_target_value, target_reward_phi, target_value_phi, \
	pred_reward, pred_value, target_policies, predicted_policies = td_data
	batch_weights, batch_indices = priority_data
	worker_reward, worker_disc_reward, worker_eps_len, evaluate_score, temperature, visit_entropy = worker_logs
	
	replay_episodes_collected = ray.get(replay_buffer.episodes_collected.remote())
	replay_buffer_size = ray.get(replay_buffer.size.remote())
	
	_msg = '#{:<10} Loss: {:<8.3f} [weighted Loss:{:<8.3f} Policy Loss: {:<8.3f} Value Loss: {:<8.3f} ' \
	       'Reward Loss: {:<8.3f} ] Replay Episodes Collected: {:<10d} Buffer Size: {:<10d} Lr: {:<8.3f}'
	_msg = _msg.format(step_count, loss, weighted_loss, policy_loss, value_loss, reward_loss,
	                   replay_episodes_collected, replay_buffer_size, lr)
	train_logger.info(_msg)
	
	if evaluate_score is not None:
		evaluate_msg = '#{:<10} evaluate Score: {:<10}'.format(step_count, evaluate_score)
		evaluate_logger.info(evaluate_msg)
	
	if summary_writer is not None:
		if config.debug:
			for name, W in model.named_parameters():
				summary_writer.add_histogram('after_grad_clip' + '/' + name + '_grad', W.grad.data.cpu().numpy(),
				                             step_count)
				summary_writer.add_histogram('network_weights' + '/' + name, W.data.cpu().numpy(), step_count)
			pass
		summary_writer.add_histogram('replay_data/replay_buffer_priorities',
		                             ray.get(replay_buffer.get_priorities.remote()),
		                             step_count)
		summary_writer.add_histogram('replay_data/batch_weight', batch_weights, step_count)
		summary_writer.add_histogram('replay_data/batch_indices', batch_indices, step_count)
		summary_writer.add_histogram('train_data_dist/target_reward', target_reward.flatten(), step_count)
		summary_writer.add_histogram('train_data_dist/target_value', target_value.flatten(), step_count)
		summary_writer.add_histogram('train_data_dist/transformed_target_reward', trans_target_reward.flatten(),
		                             step_count)
		summary_writer.add_histogram('train_data_dist/transformed_target_value', trans_target_value.flatten(),
		                             step_count)
		summary_writer.add_histogram('train_data_dist/target_reward_phi', target_reward_phi.unique().flatten(),
		                             step_count)
		summary_writer.add_histogram('train_data_dist/target_value_phi', target_value_phi.unique().flatten(),
		                             step_count)
		summary_writer.add_histogram('train_data_dist/pred_reward', pred_reward.flatten(), step_count)
		summary_writer.add_histogram('train_data_dist/pred_value', pred_value.flatten(), step_count)
		summary_writer.add_histogram('train_data_dist/pred_policies', predicted_policies.flatten(), step_count)
		summary_writer.add_histogram('train_data_dist/target_policies', target_policies.flatten(), step_count)
		
		summary_writer.add_scalar('train/loss', loss, step_count)
		summary_writer.add_scalar('train/weighted_loss', weighted_loss, step_count)
		summary_writer.add_scalar('train/policy_loss', policy_loss, step_count)
		summary_writer.add_scalar('train/value_loss', value_loss, step_count)
		summary_writer.add_scalar('train/reward_loss', reward_loss, step_count)
		summary_writer.add_scalar('train/episodes_collected', ray.get(replay_buffer.episodes_collected.remote()),
		                          step_count)
		summary_writer.add_scalar('train/replay_buffer_len', ray.get(replay_buffer.size.remote()), step_count)
		summary_writer.add_scalar('train/lr', lr, step_count)
	
		if worker_reward is not None:
			summary_writer.add_scalar('workers/reward', worker_reward, step_count)
			summary_writer.add_scalar('workers/disc_reward', worker_disc_reward, step_count)
			summary_writer.add_scalar('workers/eps_len', worker_eps_len, step_count)
			summary_writer.add_scalar('workers/temperature', temperature, step_count)
			summary_writer.add_scalar('workers/visit_entropy', visit_entropy, step_count)
		
		if evaluate_score is not None:
			summary_writer.add_scalar('train/evaluate_score', evaluate_score, step_count)
			

@ray.remote
def _evaluate(config, domain_settings, shared_storage):
	evaluate_model = config.get_uniform_network().to('cpu')
	best_evaluate_score = float('-inf')
	
	while ray.get(shared_storage.get_counter.remote()) < config.training_steps:
		if ray.get(shared_storage.get_counter.remote()) > 0 and \
		        ray.get(shared_storage.get_counter.remote()) % config.evaluate_interval == 0:
			print('Evaluation started at episode: {}'.format(ray.get(shared_storage.get_counter.remote())))
			shared_storage.add_evaluate_steps.remote(ray.get(shared_storage.get_counter.remote()))
			evaluate_model.set_weights(ray.get(shared_storage.get_weights.remote()))
			evaluate_model.eval()
			
			evaluate_score = evaluate(config, domain_settings, evaluate_model, config.evaluate_episodes, 'cpu', False)
			if evaluate_score >= best_evaluate_score:
				best_evaluate_score = evaluate_score
				print('best_evaluate_score: {}'.format(best_evaluate_score))
				torch.save(evaluate_model.state_dict(), config.model_path)
			if domain_settings['phase'] == 'transition':
				# It always saves the model in transition phase regardless of the performance
				torch.save(evaluate_model.state_dict(), config.model_path)
			
			shared_storage.add_evaluate_log.remote(evaluate_score)
			performance, _ = ray.get(shared_storage.get_evaluate_log.remote())
			time.sleep(10)
			if ray.get(shared_storage.get_counter.remote()) > 5000 and sum(performance[-100:]) / 100 > 0.98:
				break