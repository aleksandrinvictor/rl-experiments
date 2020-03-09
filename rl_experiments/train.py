from rl_experiments.common.train import train_on_policy, train_off_policy, fill_buffer
from rl_experiments.common.sampler import Sampler
from rl_experiments.common.models import *
from rl_experiments.common.utils import make_env, get_env_type, LinearDecay
from rl_experiments.common.replay_buffer import ReplayBuffer
from rl_experiments.a2c import A2C
from rl_experiments.reinforce import REINFORCE
from rl_experiments.dqn import DQN
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import click
import logging
import os
import random
import numpy as np
import hydra
from omegaconf import DictConfig


ONPOLICY_ALGS = ['reinforce', 'a2c']
OFFPOLICY_ALGS = ['dqn']


def get_model(alg_name, env, device) -> nn.Module:

    state_shape, n_actions = env.observation_space.shape, env.action_space.n

    model = None
    if alg_name == 'a2c':
        model = ActorCritic(state_shape, n_actions).to(device)
    elif alg_name == 'reinforce':
        model = MLP(state_shape, n_actions).to(device)
    elif alg_name == 'dqn':
        model = MLP(state_shape, n_actions).to(device)

    return model


def get_agent(cfg: DictConfig, model, device) -> object:
    if cfg.algorithm.name == 'a2c':
        agent = A2C(
            model,
            cfg.optimizer.learning_rate,
            cfg.algorithm.entropy_coef,
            cfg.algorithm.gamma,
            device
        )
    elif cfg.algorithm.name == 'reinforce':
        agent = REINFORCE(
            model,
            cfg.optimizer.learning_rate,
            cfg.algorithm.entropy_coef,
            cfg.algorithm.gamma,
            device
        )
    elif cfg.algorithm.name == 'dqn':
        agent = DQN(
            cfg.algorithm.type,
            model,
            cfg.algorithm.refresh_target_network_freq,
            cfg.algorithm.start_epsilon,
            cfg.algorithm.gamma,
            cfg.optimizer.learning_rate,
            cfg.algorithm.max_grad_norm,
            device
        )
    return agent


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig) -> None:

    print(cfg.pretty())

    # Init Tensorboard writer
    writer = SummaryWriter(cfg.common.output_path)

    # Set seed
    random.seed(cfg.common.seed)
    np.random.seed(cfg.common.seed)
    torch.manual_seed(cfg.common.seed)

    logger.info('Starting experiment:')
    logger.info(f'Algorithm: {cfg.algorithm.name}')
    logger.info(f'Environment: {cfg.env.name}')
    logger.info(f'Seed: {cfg.common.seed}')

    # Setup environment
    env = make_env(cfg.env.name, cfg.common.seed)
    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    logger.info(f'State shape: {state_shape}')
    logger.info(f'n actions: {n_actions}')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    sampler = Sampler(env, n_steps=cfg.train_spec.rollout_n_steps)

    model = get_model(cfg.algorithm.name, env, device)

    agent = get_agent(cfg, model, device)

    if cfg.algorithm.name in ONPOLICY_ALGS:
        train_on_policy(
            cfg.env.name,
            sampler,
            agent,
            cfg.train_spec.total_steps,
            cfg.train_spec.eval_frequency
        )
    elif cfg.algorithm.name in OFFPOLICY_ALGS:
        # Init replay buffer
        prioritized_beta_tracker = None
        if cfg.algorithm.prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(
                cfg.algorithm.replay_buffer_size,
                cfg.algorithm.prioritized_replay_alpha
            )
            prioritized_beta_tracker = LinearDecay(
                cfg.algorithm.prioritized_replay_beta_start,
                cfg.algorithm.prioritized_replay_beta_end,
                cfg.algorithm.prioritized_replay_beta_iters
            )
        else:
            replay_buffer = ReplayBuffer(cfg.algorithm.replay_buffer_size)

        replay_buffer = fill_buffer(
            sampler,
            agent,
            replay_buffer,
            cfg.algorithm.replay_buffer_start_size
        )

        # Init epsilon tracker
        eps_tracker = LinearDecay(
            cfg.algorithm.start_epsilon,
            cfg.algorithm.end_epsilon,
            cfg.algorithm.eps_iters
        )

        train_off_policy(
            cfg.env.name,
            sampler,
            agent,
            cfg.train_spec.total_steps,
            cfg.train_spec.eval_frequency,
            eps_tracker,
            replay_buffer,
            cfg.algorithm.replay_batch_size
        )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
