#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl.network import *
from deep_rl.component import *
from deep_rl.utils import *
from deep_rl.agent.BaseAgent import BaseAgent, BaseActor
from collections import deque


class Actor(BaseActor):
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
                logits, _ = self._network(np.expand_dims(config.state_normalizer(self._state), axis=0), 'both')
        else:
            logits, _ = self._network(np.expand_dims(config.state_normalizer(self._state), axis=0), 'both')

        dist = torch.distributions.Categorical(logits=logits)
        
        action = to_np(dist.sample())[0]
        
        # if self._total_steps < config.exploration_steps or np.random.rand() < config.random_action_prob():
        #     action = np.random.randint(0, len(q_values))
        # else:
        #     action = to_np(torch.argmax(dist.probs, dim=1))[0]

        next_state, reward, done, info = self._task.step(action)

        if config.bootstrap_from_timeout is True and info['TimeLimit.truncated'] is True:
            done = False
        entry = [self._state, action, reward, next_state, int(done), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class QPolicyImitator(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = Actor(config)

        self.network = config.network
        if config.use_target_network:
            self.target_network = config.network_fn()
            self.target_network.load_state_dict(self.network.state_dict())
        else:
            self.target_network = self.network
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        
        self.recent_300_returns = deque(maxlen=30)
        self.recent_3_returns = deque(maxlen=3)
        self.batch_indices = range_tensor(self.replay.batch_size)

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        # q = self.network(state, 'q values')
        # action = to_np(q.argmax(-1))
        logits = self.network(np.expand_dims(state, axis=0), 'actor')
        dist = torch.distributions.Categorical(logits=logits)
        action = to_np(torch.argmax(dist.probs, dim=1))[0]
        # action = dist.sample()
        # action = to_np(action)[0]
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        report = {}
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            # self.record_online_return(info)
            ret = info['episodic_return']
            if ret is not None:
                report.setdefault('episodic_return', []).append(ret)
                self.recent_300_returns.append(ret)
                self.recent_3_returns.append(ret)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            # Q learning
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next = self.target_network(next_states, 'q values').detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states, 'q values'), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q = self.network(states, 'q values')
            q_actions = q[self.batch_indices, actions]
            q_loss = (q_next - q_actions).pow(2).mul(0.5).mean()

            # # Update
            # self.optimizer.zero_grad()
            # q_loss.backward()
            # nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            # if hasattr(config, "async_actor") and config.async_actor is True:
            #     with config.lock:
            #         self.optimizer.step()
            # else:
            #     self.optimizer.step()
            
            # Imitation Learning
            experiences = self.replay.sample()
            states, _, _, _, _ = experiences
            states = self.config.state_normalizer(states)
            logits, q = self.network(states, 'both')
            # logits = self.network(states, 'actor')
            max_q_actions = torch.argmax(q, dim=1)
            actor_criterion = nn.CrossEntropyLoss()
            actor_loss = actor_criterion(logits, max_q_actions)
            actor_actions = logits.max(1)[1]
            dist = torch.distributions.Categorical(logits=logits)
            entropy_loss = dist.entropy().unsqueeze(-1).mean()
            
            report.setdefault('q_values', []).append(to_np(q).mean())
            report.setdefault('actor_loss', []).append(to_np(actor_loss))
            report.setdefault('q_a_prob', []).append(to_np(dist.probs[self.batch_indices, max_q_actions]))
            report.setdefault('q_loss', []).append(to_np(q_loss))
            report.setdefault('q_actor_matching', []).append(1 - np.mean(np.abs(to_np(actor_actions - max_q_actions))))
            report.setdefault('entropy_loss', []).append(to_np(entropy_loss))
            
            # actor_weight = config.actor_weight * np.exp((np.mean(self.recent_300_returns) - np.mean(self.recent_3_returns)) / config.actor_tau)
            # report.setdefault('actor_weight', []).append(actor_weight)
            
            # print(actor_weight)
            q_actor_matching_square = (1 - np.mean(np.abs(to_np(actor_actions - max_q_actions)))) ** 2
            loss = q_loss + config.actor_weight * q_actor_matching_square * actor_loss - config.entropy_weight * entropy_loss
            # print(logits, q_loss, actor_loss, entropy_loss)
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            if hasattr(config, "async_actor") and config.async_actor is True:
                with config.lock:
                    self.optimizer.step()
            else:
                self.optimizer.step()

        if config.use_target_network and self.total_steps / self.config.sgd_update_frequency % self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            
        return report