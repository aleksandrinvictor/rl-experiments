from a2c import A2C
from torch.utils.tensorboard import SummaryWriter
from rl_experiments.common.models import ActorCritic
from rl_experiments.common.utils import make_env, get_env_type
from rl_experiments.common.sampler import Sampler
from typing import Dict, Any, List, NoReturn

import gym
import numpy as np
import torch
import torch.nn as nn
import click
import logging
import os
import random


def train(
    env_name: str,
    sampler: Sampler,
    agent: A2C,
    total_steps: int,
    eval_frequency: int
) -> NoReturn:

    num_epochs = int(total_steps // sampler.n_steps) + 1
    for epoch in range(num_epochs):
        trajectory: Dict[str, List[Any]] = sampler.sample(agent)

        agent.update(trajectory)

        if epoch % eval_frequency == 0:
            # Eval the agent
            eval_reward = evaluate(
                make_env(env_name, seed=int(epoch)),
                agent,
                t_max=10000
            )
            print(
                f'step: {epoch}, mean_reward_per_episode: {eval_reward}'
            )


def evaluate(env, agent, t_max=10000):
    """ Plays n_games full games. Returns mean reward. """
    s = env.reset()
    total_reward = 0
    n_episodes = 0
    episode_reward = 0
    for _ in range(t_max):
        action = agent.get_actions(np.array([s]))
        s, r, done, _ = env.step(action)
        episode_reward += r
        if done:
            s = env.reset()
            n_episodes += 1
            total_reward += episode_reward
            episode_reward = 0

    if n_episodes == 0:
        return episode_reward

    return total_reward / n_episodes


@click.command()
@click.option('-e', '--env_name', type=str, default='CartPole-v1')
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
    model = ActorCritic(state_shape, n_actions)

    agent = A2C(
        model,
        learning_rate,
        entropy_coef,
        gamma,
        device
    )

    train(env_name, sampler, agent, total_steps, eval_frequency)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()