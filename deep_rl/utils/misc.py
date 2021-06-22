#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import os
import time
from pathlib import Path
import copy
import inspect
from ..utils.logger import get_logger


def run_steps(config, muzero_config, domain_settings, summary_writer):
    config.seed = config.sweep_id * config.num_workers + config.rank
    agent = config.agent_class(config, muzero_config, domain_settings, summary_writer)
    agent_name = agent.__class__.__name__
    if config.rank == 0:
        agent.logger = get_logger(config.exp_name, config.sweep_id, log_level=config.log_level)
        report_string = "\n"
        for k in config.param_sweeper_dict:
            report_string += "%s: %s \n" % (str(k), str(config.param_sweeper_dict[k]))
        agent.logger.info(report_string)
    total_episodes = 0
    reports = {}
    t0 = time.time()
    last_total_steps = 0
    while True:
        if config.rank == 0:
            report = agent.step()
            # if report is not None:
            #     for key in report:
            #         reports.setdefault(key, []).extend(report[key])
            # log_flag = int(agent.total_steps / config.log_interval) > int(last_total_steps / config.log_interval)
            # save_network_flag = int(agent.total_steps / config.save_interval) > int(last_total_steps / config.save_interval)
            # eval_flag = int(agent.total_steps / config.eval_interval) > int(last_total_steps / config.eval_interval)
            # if log_flag and len(reports) != 0:
            #     report_string = '\ntotal steps %d\n' % (agent.total_steps)
            #     for report_name in reports:
            #         report = reports[report_name]
            #         if report_name == 'episodic_return':
            #             total_episodes += len(reports[report_name])
            #             report_string += 'total episodes %3d\n' % (total_episodes)
            #         if report_name == 'rewards':
            #             report_string += 'average reward %3f\n' % (np.sum(report) / config.log_interval)
            #         if len(report) != 0:
            #             report_string += report_name + ' %.3f/%.3f/%.3f/%.3f/%d (mean/median/min/max/num)\n' % (
            #                 np.mean(report), np.median(report), np.min(report),
            #                 np.max(report), len(report)
            #             )
            #     report_string += '%.3f steps/s\n' % (config.log_interval / (time.time() - t0))
            #     agent.logger.info(report_string)
            #     reports = {}
            # t0 = time.time()
            # if config.if_save_network and save_network_flag:
            #     agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
            # if config.if_eval_episodes and eval_flag:
            #     episodic_returns, episodic_lengths = agent.eval_episodes()
            #     report_string = '\nepisodic_return_test' + ' %.3f/%.3f/%.3f/%.3f/%d (mean/median/min/max/num)\n' % (
            #         np.mean(episodic_returns), np.median(episodic_returns),
            #         np.min(episodic_returns), np.max(episodic_returns), len(episodic_returns)
            #     )
            #     report_string += 'episodic_length_test' + ' %.3f/%.3f/%.3f/%.3f/%d (mean/median/min/max/num)\n' % (
            #         np.mean(episodic_lengths), np.median(episodic_lengths),
            #         np.min(episodic_lengths), np.max(episodic_lengths), len(episodic_lengths)
            #     )
            #     agent.logger.info(report_string)
            # if config.if_eval_steps and eval_flag:
            #     avg_reward = agent.eval_n_steps()
            #     report_string = '\naverage_reward_test' + ' %.3f over %d steps\n' % (avg_reward, config.eval_steps)
            #     agent.logger.info(report_string)
        else:
            agent.step()
            
        # if agent.total_steps >= config.max_steps:
        #     agent.close()
        #     break
        #
        # last_total_steps = agent.total_steps

# def get_default_log_dir(name):
#     return './log/%s-%s' % (name, get_time_str())


def get_args(func):
    signature = inspect.signature(func)
    return [k for k, v in signature.parameters.items()]

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def shallow_copy(obj):
    return copy.copy(obj)


def deep_copy(obj):
    return copy.deepcopy(obj)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def generate_tag(params):
    if 'tag' in params.keys():
        return
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    del params['game']
    del params['run']
    str = ['%s_%s' % (k, v) for k, v in sorted(params.items())]
    tag = '%s-%s-run-%d' % (game, '-'.join(str), run)
    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern):
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
        
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count