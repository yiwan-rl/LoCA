from deep_rl.component import Env
from deep_rl.component.replay import AsyncReplay, Replay
from deep_rl.network import network_bodies
from deep_rl.network.network_bodies import NatureConvBody, FCBody
from deep_rl.network.network_heads import VanillaNet, CategoricalActorCriticNet, CategoricalActorQCriticNet, \
    QLearningOptionCriticNet, StochasticOptionCriticNet, StochasticOptionCriticNetNoShare,\
	ValueComplexityOptionCriticNet, OptionValueLearningNet
from deep_rl.utils.schedule import LinearSchedule
from deep_rl import agent
from deep_rl.utils import normalizer
from deep_rl.utils.misc import run_steps
from deep_rl.utils.torch_utils import random_seed, set_one_thread
from deep_rl.utils.param_config import ParamConfig

import torch
import torch.nn as nn
import os
import time
import argparse
import torch.multiprocessing as mp
from alphaex.sweeper import Sweeper

from LoCA_MountainCar.utils import create_filename, save_results, update_summary_writer
from LoCA_MountainCar.config import get_domain_setting, get_experiment_setting
from deep_rl.agent.muzero_helpers import init_logger, make_results_dir
from torch.utils.tensorboard import SummaryWriter

os.environ["SDL_VIDEODRIVER"] = "dummy"


def set_task_fn(cfg):
    cfg.task_fn = lambda seed: Env(cfg.task_name, cfg.task_args, seed=seed, timeout=cfg.timeout)
    cfg.eval_task = cfg.task_fn(seed=0)

    cfg.observation_space = cfg.eval_task.observation_space
    cfg.action_space = cfg.eval_task.action_space


def set_replay_fn(cfg):
    if cfg.async_replay:
        cfg.replay_fn = lambda: AsyncReplay(memory_size=int(cfg.memory_size), batch_size=cfg.batch_size)
    else:
        cfg.replay_fn = lambda: Replay(memory_size=int(cfg.memory_size), batch_size=cfg.batch_size)


def set_network_fn_and_network(cfg, muzero_config):
    # load parameters if use a pretrained network
    # if cfg.load:
    #     saved_state = torch.load('{0}{1}.dat'.format(args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
    #     cfg.shared_network.load_state_dict(saved_state)

    # input_dim = cfg.observation_space.shape[0]
    input_dim = cfg.state_normalizer.dim
    action_dim = cfg.action_space.n
    
    if cfg.network_name == 'ActionValue':
        q_body_class = getattr(network_bodies, cfg.q_body)
        cfg.network_fn = lambda: VanillaNet(action_dim, q_body_class(
                input_dim, hidden_units=tuple(cfg.q_body_network), gate=getattr(torch, cfg.q_body_gate)
            ), bias=cfg.output_layer_bias, initialization=cfg.initialization)
    elif cfg.network_name == 'IntraOptionTDLearning':
        q_body_class = getattr(network_bodies, cfg.q_body)
        cfg.network_fn = lambda: OptionValueLearningNet(False, action_dim, q_body_class(
                input_dim, hidden_units=tuple(cfg.q_body_network), gate=getattr(torch, cfg.q_body_gate)
            ), bias=cfg.output_layer_bias, initialization=cfg.initialization)
    elif cfg.network_name == 'InterOptionQLearning':
        q_body_class = getattr(network_bodies, cfg.q_body)
        cfg.network_fn = lambda: OptionValueLearningNet(True, action_dim, q_body_class(
                input_dim, hidden_units=tuple(cfg.q_body_network), gate=getattr(torch, cfg.q_body_gate)
            ), bias=cfg.output_layer_bias, initialization=cfg.initialization)
    elif cfg.network_name == 'ActorCritic':
        actor_body_class = getattr(network_bodies, cfg.actor_body)
        critic_body_class = getattr(network_bodies, cfg.critic_body)
        cfg.network_fn = lambda: CategoricalActorCriticNet(
            input_dim, action_dim,
            actor_body=actor_body_class(
                input_dim, hidden_units=tuple(cfg.actor_body_network), gate=getattr(torch, cfg.actor_body_gate)
            ),
            critic_body=critic_body_class(
                input_dim, hidden_units=tuple(cfg.critic_body_network), gate=getattr(torch, cfg.critic_body_gate)
            )
        )
    elif cfg.network_name == 'ActorQCritic':
        cfg.network_fn = lambda: CategoricalActorQCriticNet(
            input_dim, action_dim, actor_body=FCBody(input_dim), critic_body=FCBody(input_dim)
        )
    elif cfg.network_name == 'QLearningOptionCritic':
        cfg.network_fn = lambda: QLearningOptionCriticNet(FCBody(input_dim), action_dim, cfg.num_options)
    elif cfg.network_name == 'StochasticOptionCritic':
        phi_body_class = getattr(network_bodies, cfg.phi_body)
        pi_body_class = getattr(network_bodies, cfg.pi_body)
        beta_body_class = getattr(network_bodies, cfg.beta_body)
        mu_body_class = getattr(network_bodies, cfg.mu_body)
        q_body_class = getattr(network_bodies, cfg.q_body)
        
        cfg.network_fn = lambda: StochasticOptionCriticNet(
            state_dim=input_dim, action_dim=action_dim, num_options=cfg.num_options,
            phi_body=phi_body_class(input_dim, hidden_units=tuple(cfg.phi_body_network), gate=torch.tanh),
            pi_body=pi_body_class(input_dim, hidden_units=tuple(cfg.pi_body_network), gate=torch.tanh),
            beta_body=beta_body_class(input_dim, hidden_units=tuple(cfg.beta_body_network), gate=torch.tanh),
            mu_body=mu_body_class(input_dim, hidden_units=tuple(cfg.mu_body_network), gate=torch.tanh),
            q_body=q_body_class(input_dim, hidden_units=tuple(cfg.q_body_network), gate=torch.tanh)
        )
    elif cfg.network_name == 'StochasticOptionCriticNoShare':
        phi_body_class = getattr(network_bodies, cfg.phi_body)
        pi_body_class = getattr(network_bodies, cfg.pi_body)
        beta_body_class = getattr(network_bodies, cfg.beta_body)
        mu_body_class = getattr(network_bodies, cfg.mu_body)
        q_body_class = getattr(network_bodies, cfg.q_body)
        
        if cfg.critic_input == 'xy':
            critic_input_dim = input_dim
        elif cfg.critic_input == 'tabular':
            critic_input_dim = cfg.eval_task.env.tabular_dim
        else:
            raise NotImplementedError
    
        cfg.network_fn = lambda: StochasticOptionCriticNetNoShare(
            state_dim=input_dim, action_dim=action_dim, num_options=cfg.num_options,
            phi_body=phi_body_class(
                input_dim, hidden_units=tuple(cfg.phi_body_network),
                gate=getattr(torch, cfg.phi_body_gate), return_tc=cfg.return_tc
            ),
            pi_bodies=nn.ModuleList(
                [pi_body_class(
                    input_dim, hidden_units=tuple(cfg.pi_body_network),
                    gate=getattr(torch, cfg.pi_body_gate), return_tc=cfg.return_tc
                ) for _ in range(cfg.num_options)]
            ),
            beta_bodies=nn.ModuleList(
                [beta_body_class(
                    input_dim, hidden_units=tuple(cfg.beta_body_network),
                    gate=getattr(torch, cfg.beta_body_gate), return_tc=cfg.return_tc
                ) for _ in range(cfg.num_options)]
            ),
            mu_body=mu_body_class(
                input_dim, hidden_units=tuple(cfg.mu_body_network),
                gate=getattr(torch, cfg.mu_body_gate), return_tc=cfg.return_tc
            ),
            q_bodies=nn.ModuleList(
                [q_body_class(
                    critic_input_dim, hidden_units=tuple(cfg.q_body_network),
                    gate=getattr(torch, cfg.q_body_gate), return_tc=cfg.return_tc
                ) for _ in range(cfg.num_options)]
            ),
        )
    elif cfg.network_name == 'ValueComplexityOptionCritic':
        phi_body_class = getattr(network_bodies, cfg.phi_body)
        pi_body_class = getattr(network_bodies, cfg.pi_body)
        beta_body_class = getattr(network_bodies, cfg.beta_body)
        mu_body_class = getattr(network_bodies, cfg.mu_body)
        q_body_class = getattr(network_bodies, cfg.q_body)
        
        if cfg.critic_input == 'xy':
            critic_input_dim = input_dim
        elif cfg.critic_input == 'tabular':
            critic_input_dim = cfg.eval_task.env.tabular_dim
        else:
            raise NotImplementedError
            
        cfg.network_fn = lambda: ValueComplexityOptionCriticNet(
            state_dim=input_dim,
            action_dim=action_dim,
            num_options=cfg.num_options,
            cost_weight=cfg.cost_weight,
            mu_entropy_weight=cfg.mu_entropy_weight,
            pi_entropy_weight=cfg.pi_entropy_weight,
            beta_entropy_weight=cfg.beta_entropy_weight,
            phi_body=phi_body_class(
                input_dim, hidden_units=tuple(cfg.phi_body_network),
                gate=getattr(torch, cfg.phi_body_gate), dropout_rate=cfg.dropout_rate, return_cost=cfg.return_cost,
                lamba=cfg.lamba,weight_decay=cfg.weight_decay
            ),
            pi_bodies=nn.ModuleList(
                [pi_body_class(
                    input_dim, hidden_units=tuple(cfg.pi_body_network),
                    gate=getattr(torch, cfg.pi_body_gate), dropout_rate=cfg.dropout_rate, return_cost=cfg.return_cost,
                    lamba=cfg.lamba, weight_decay=cfg.weight_decay
                ) for _ in range(cfg.num_options)]
            ),
            beta_bodies=nn.ModuleList(
                [beta_body_class(
                    input_dim, hidden_units=tuple(cfg.beta_body_network),
                    gate=getattr(torch, cfg.beta_body_gate), dropout_rate=cfg.dropout_rate, return_cost=cfg.return_cost,
                    lamba=cfg.lamba, weight_decay=cfg.weight_decay
                ) for _ in range(cfg.num_options)]
            ),
            mu_body=mu_body_class(
                input_dim, hidden_units=tuple(cfg.mu_body_network),
                gate=getattr(torch, cfg.mu_body_gate), dropout_rate=cfg.dropout_rate, return_cost=cfg.return_cost,
                lamba=cfg.lamba, weight_decay=cfg.weight_decay
            ),
            q_bodies=nn.ModuleList(
                [q_body_class(
                    critic_input_dim, hidden_units=tuple(cfg.q_body_network), gate=getattr(torch, cfg.q_body_gate)
                ) for _ in range(cfg.num_options)]
            ),
            return_cost=cfg.return_cost,
            lamba=cfg.lamba,
            weight_decay=cfg.weight_decay
        )
    elif cfg.network_name == 'MuZero':
        cfg.network_fn = muzero_config.get_uniform_network
        # model.train()
    else:
        raise NotImplementedError
    cfg.network = cfg.network_fn()
    cfg.network = cfg.network.to(muzero_config.device)
    cfg.network.train()
    if cfg.num_workers > 1:
        cfg.network.share_memory()


def set_random_action_prob(cfg):
    cfg.random_action_prob = LinearSchedule(
        cfg.linear_schedule_start, cfg.linear_schedule_end, cfg.linear_schedule_steps
    )
    cfg.random_option_prob = LinearSchedule(
        cfg.linear_schedule_start, cfg.linear_schedule_end, cfg.linear_schedule_steps
    )


def set_normalizer_class(cfg):
    normalizer_class = getattr(normalizer, cfg.state_normalizer_name)
    cfg.state_normalizer = normalizer_class(cfg)
    normalizer_class =  getattr(normalizer, cfg.reward_normalizer_name)
    cfg.reward_normalizer = normalizer_class(cfg)


def set_agent_class(cfg):
    cfg.agent_class = getattr(agent, cfg.agent_name)


if __name__ == '__main__':
    from LoCA_MountainCar.muzero.env import muzero_config as muzero_cfg
    
    ########################################################
    muzero_cfg.flippedTask = True
    muzero_cfg.flippedActions = True
    ########################################################
    
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies run number and configuration')
    parser.add_argument('--config-file')
    # parser.add_argument('--run', default=0, type=int)
    parser.add_argument('--method', default='sarsa_lambda', help='Name of the method')
    parser.add_argument('--env', default='MountainCar', help='Name of the environment')
    parser.add_argument('--no_pre_training', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False, help='load previous pre-trained agents')
    parser.add_argument('--save', action='store_true', default=True, help='save the agent and the results')
    parser.add_argument('--flipped_terminals', action='store_true', default=False, help='flip the rewards associated '
                                                                                        'with terminal 1 and terminal 2')
    parser.add_argument('--flipped_actions', action='store_true', default=False, help='Shuffle the actions to cancel '
                                                                                      'the effect of model learning')

    args = parser.parse_args()

    ########################################################
    experiment_settings = get_experiment_setting(args)
    domain_settings = get_domain_setting(args)
    filename = create_filename(args)
    print("file: ", filename)
    experiment_settings['filename'] = filename
    exp_path = muzero_cfg.set_config(muzero_cfg, domain_settings)
    exp_path, log_base_path = make_results_dir(exp_path, muzero_cfg)
    # muzero_cfg.seed = args.seed
    # muzero_cfg.opr = args.opr
    # set-up logger
    init_logger(log_base_path)
    summary_writer = SummaryWriter(exp_path, flush_secs=10)
    ########################################################
    
    # exp_name = '/'.join(args.config_file.split('/')[:-1])
    exp_name = args.config_file.split('/')[1]
    project_root = os.path.abspath(os.path.dirname(__file__))
    param_sweeper = Sweeper(os.path.join(project_root, args.config_file))
    param_sweeper_dict = param_sweeper.parse(args.id)
    
    # logger = get_logger(exp_name, args.id, log_level=0)
    # for k in param_sweeper_dict:
    #     print(k, param_sweeper_dict[k])
        # logger.info(k + ": "+ str(param_sweeper_dict[k]))
    cfg = ParamConfig(param_sweeper_dict)
    cfg.param_sweeper_dict = param_sweeper_dict
    cfg.exp_name = exp_name
    cfg.data_root = os.path.join(project_root, 'data', 'output')
    cfg.sweep_id = args.id
    
    set_one_thread()
    random_seed(cfg.sweep_id)

    # Setting up the config
    set_task_fn(cfg)

    set_normalizer_class(cfg)
    
    set_network_fn_and_network(cfg, muzero_cfg)

    # set_optimizer_fn(cfg)

    set_replay_fn(cfg)

    set_random_action_prob(cfg)

    set_agent_class(cfg)

    # make output directory
    bash_script = "mkdir -p experiment/%s/outputs" % cfg.exp_name
    print(bash_script)
    myCmd = os.popen(bash_script).read()

    # start processes in different cores of cpu
    assert(cfg.num_workers > 0)
    processes = []

    if cfg.num_workers == 1:
        cfg.rank = 0
        # cfg.logger = logger
        run_steps(cfg, muzero_cfg, domain_settings, summary_writer)
    else:
        for rank in reversed(range(0, cfg.num_workers)):
            cfg.rank = rank
            # if rank == 0:
                # cfg.logger = logger
            p = mp.Process(target=run_steps, args=(cfg, muzero_cfg, domain_settings, summary_writer))
            p.start()
            processes.append(p)
            time.sleep(0.1)

        for p in processes:
            time.sleep(0.1)
            p.join()
