#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class ActorCritic(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn(config.seed)
        self.network = config.network
        self.optimizer = set_optimizer(self.network.parameters(), config)
        self.total_steps = 0
        self.states = self.task.reset()
    
    def eval_step(self, ep, state, reward, done, info):
        if done:
            state = self.config.eval_env.env.reset_test()
        prediction = self.network(np.expand_dims(self.config.state_normalizer(state), axis=0))
        return to_np(prediction['a']).item()

    def step(self):
        report = {'episodic_return': [], 'actor_loss': [], 'critic_loss': [], 'max_a_prob': []}
        config = self.config
        storage = Storage(config.rollout_length)
        for _ in range(config.rollout_length):
            prediction = self.network(np.expand_dims(config.state_normalizer(self.states), axis=0))
            next_states, reward, terminal, info = self.task.step(to_np(prediction['a']).item())
            # self.record_online_return(info)
            ret = info['episodic_return']
            if ret is not None:
                report['episodic_return'].append(ret)
                report.setdefault('episodic_length', []).append(info['episodic_length'])
            report['max_a_prob'].append(to_np(prediction['max_a_prob']).item())
            reward = config.reward_normalizer(reward)
            storage.add(prediction)
            storage.add({'r': tensor(reward), 'm': tensor(1 - terminal)})
            self.states = next_states
            self.total_steps += 1

        prediction = self.network(np.expand_dims(config.state_normalizer(self.states), axis=0))
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((1, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        entropy_loss = entropy.mean()

        report['actor_loss'].append(to_np(policy_loss))
        report['critic_loss'].append(to_np(value_loss))

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss + config.value_weight * value_loss).backward()
        if self.config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        return report