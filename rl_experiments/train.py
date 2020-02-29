from rl_experiments.common.train import train_on_policy
from rl_experiments.common.sampler import Sampler
from rl_experiments.common.models import *
from rl_experiments.common.utils import make_env, get_env_type
from rl_experiments.a2c import A2C
from rl_experiments.reinforce import REINFORCE
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import click
import logging
import os
import random
import numpy as np


def get_model(alg_name, env, device) -> nn.Module:

    state_shape, n_actions = env.observation_space.shape, env.action_space.n

    model = None
    if alg_name == 'a2c':
        model = ActorCritic(state_shape, n_actions).to(device)
    elif alg_name == 'reinforce':
        model = MLP(state_shape, n_actions).to(device)

    return model


@click.command()
@click.option('-e', '--env_name', type=str, default='CartPole-v1')
@click.option('-a', '--alg_name', type=str, default='reinforce')
@click.option('-t', '--total_steps', type=float, default=9*10**5)
@click.option('-n_steps', '--rollout_n_steps', type=int, default=300)
@click.option('-gamma', '--gamma', type=float, default=0.99)
@click.option('-lr', '--learning_rate', type=float, default=3e-4)
@click.option('-entropy', '--entropy_coef', type=float, default=1e-3)
@click.option('-eval', '--eval_frequency', type=int, default=100)
@click.option('-o', '--output_path', type=str, default='./runs')
@click.option('-seed', '--seed', type=int, default=42)
def main(
    env_name: str,
    alg_name: str,
    total_steps: int,
    rollout_n_steps: int,
    gamma: float,
    learning_rate: float,
    entropy_coef: float,
    eval_frequency: int,
    output_path: str,
    seed: int
):

    # Add hyperparameters to tensorboard
    hparams = {key: val for (key, val) in locals().items() if val is not None}
    writer = SummaryWriter(output_path)
    writer.add_hparams(hparams, {})

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info('Starting experiment:')
    logger.info(f'Algorithm: {alg_name}')
    logger.info(f'Environment: {env_name}')
    logger.info(f'Seed: {seed}')

    # Setup environment
    env = make_env(env_name, seed)
    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    logger.info(f'State shape: {state_shape}')
    logger.info(f'n actions: {n_actions}')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    sampler = Sampler(env, n_steps=rollout_n_steps)

    model = get_model(alg_name, env, device)

    if alg_name == 'a2c':
        agent = A2C(
            model,
            learning_rate,
            entropy_coef,
            gamma,
            device
        )
    elif alg_name == 'reinforce':
        agent = REINFORCE(
            model,
            learning_rate,
            entropy_coef,
            gamma,
            device
        )

    train_on_policy(
        env_name,
        sampler,
        agent,
        total_steps,
        eval_frequency
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
