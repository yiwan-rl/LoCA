import logging
import os
import shutil
import numpy as np
from scipy.stats import entropy
import ray
from torch.nn import L1Loss
from .mcts import *


def make_results_dir(exp_path, args):
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError('{} is not empty. Please use --force to overwrite it'.format(exp_path))
        else:
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    return exp_path, log_path


def init_logger(base_path):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def select_action(node, temperature=1, deterministic=True):
    visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i, _ in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        action_pos = np.argmax([v for v, _ in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return visit_counts[action_pos][1], count_entropy


@ray.remote
class SharedStorage(object):
    def __init__(self, model):
        self.step_counter = 0
        self.model = model
        self.reward_log = []
        self.disc_reward_log = []
        self.evaluate_log = []
        self.evaluate_log_all, self.evaluate_steps = [], []
        self.eps_lengths = []
        self.temperature_log = []
        self.visit_entropies_log = []

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def set_data_worker_logs(self, eps_len, eps_reward, disc_reward, temperature, visit_entropy):
        self.eps_lengths.append(eps_len)
        self.reward_log.append(eps_reward)
        self.disc_reward_log.append(disc_reward)
        self.temperature_log.append(temperature)
        self.visit_entropies_log.append(visit_entropy)

    def add_evaluate_log(self, score):
        self.evaluate_log.append(score)
        self.evaluate_log_all.append(score)

    def add_evaluate_steps(self, steps):
        self.evaluate_steps.append(steps)

    def get_evaluate_log(self):
        return self.evaluate_log_all, self.evaluate_steps

    def get_worker_logs(self):
        if len(self.reward_log) > 0:
            reward = sum(self.reward_log) / len(self.reward_log)
            disc_reward = sum(self.disc_reward_log) / len(self.disc_reward_log)
            eps_lengths = sum(self.eps_lengths) / len(self.eps_lengths)
            temperature = sum(self.temperature_log) / len(self.temperature_log)
            visit_entropy = sum(self.visit_entropies_log) / len(self.visit_entropies_log)

            self.reward_log = []
            self.eps_lengths = []
            self.disc_reward_log = []
            self.temperature_log = []
            self.visit_entropies_log = []

        else:
            reward = None
            disc_reward = None
            eps_lengths = None
            temperature = None
            visit_entropy = None

        if len(self.evaluate_log) > 0:
            evaluate_score = sum(self.evaluate_log) / len(self.evaluate_log)
            self.evaluate_log = []
        else:
            evaluate_score = None

        return reward, disc_reward, eps_lengths, evaluate_score, temperature, visit_entropy


@ray.remote
class DataWorker(object):
    def __init__(self, rank, config, domain_settings, shared_storage, replay_buffer):
        self.rank = rank
        self.config = config
        self.domain_settings = domain_settings
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer

    def run(self):
        model = self.config.get_uniform_network()
        with torch.no_grad():
            while ray.get(self.shared_storage.get_counter.remote()) < self.config.training_steps and self.shared_storage is not None:
                model.set_weights(ray.get(self.shared_storage.get_weights.remote()))
                model.eval()
                env = self.config.new_game(self.domain_settings, self.config.seed + self.rank)

                if env.phase == 'train':
                    env.set_task(0)
                    env.set_phase('train')
                elif env.phase == 'transition':
                    env.set_task(1)
                    env.set_phase('transition')
                elif env.phase == 'test':
                    env.set_task(1)
                    env.set_phase('test')

                print('Init state range: ', env.init_state)
                obs = env.reset()
                terminal = 0
                G, total_discount = 0, 1
                priorities = []
                eps_reward, eps_steps, visit_entropies = 0, 0, 0
                trained_steps = ray.get(self.shared_storage.get_counter.remote())
                _temperature = self.config.visit_softmax_temperature_fn(num_moves=len(env.history),
                                                                        trained_steps=trained_steps)
                while terminal == 0 and eps_steps <= self.config.max_moves:
                    root = Node(0)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                    network_output = model.initial_inference(obs)
                    root.expand(env.to_play(), env.legal_actions(), network_output)
                    root.add_exploration_noise(dirichlet_alpha=self.config.root_dirichlet_alpha,
                                               exploration_fraction=self.config.root_exploration_fraction)
                    MCTS(self.config).run(root, env.action_history(), model)
                    action, visit_entropy = select_action(root, temperature=_temperature, deterministic=False)
                    obs, reward, terminal, info = env.step(action.index)
                    env.store_search_statistics(root)

                    eps_reward += reward
                    G += total_discount * reward
                    total_discount *= env.gamma
                    eps_steps += 1
                    visit_entropies += visit_entropy

                    if not self.config.use_max_priority:
                        error = L1Loss(reduction='none')(network_output.value,
                                                         torch.tensor([[root.value()]])).item()
                        priorities.append(error + 1e-5)

                performance, _ = ray.get(self.shared_storage.get_evaluate_log.remote())

                if ray.get(self.shared_storage.get_counter.remote()) > 5000 and sum(performance[-100:]) / 100 > 0.98:
                    break

                print('Terminal Ended:{}, reward: {}, Init state: {}'.format(terminal, eps_reward, env.init))
                env.close()
                self.replay_buffer.save_game.remote(env,
                                                    priorities=None if self.config.use_max_priority else priorities)
                # Todo: refactor with env attributes to reduce variables
                visit_entropies /= eps_steps
                self.shared_storage.set_data_worker_logs.remote(eps_steps, eps_reward, G, _temperature, visit_entropies)