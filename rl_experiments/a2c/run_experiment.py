from a2c import A2C
from torch.utils.tensorboard import SummaryWriter
from rl_experiments.common.models import MLP, Cnn, MLP_test
from rl_experiments.common.utils import make_env, get_env_type
from rl_experiments.common.sampler import Sampler

import gym
import numpy as np
import torch
import torch.nn as nn
import click
import logging
import os
import random


def train(
    sampler: Sampler,
    agent,
    num_epochs: int = 100
):

    for epoch in range(10000):
        trajectories = sampler.sample(agent)
        agent.update(trajectories)

        if epoch % 100 == 0:
            # Eval the agent
            eval_reward = evaluate(
                make_env('CartPole-v1', seed=int(epoch)),
                agent,
                t_max=10000
            )
            print(
                f'step: {epoch}, mean_reward_per_episode: {eval_reward}'
            )


def evaluate(env, agent, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
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


def main():
    # print(np.random.choice(3, p=[0.9, 0.01, 0.09]))
    env = make_env('CartPole-v1', 42)
    state_shape, n_actions = env.observation_space.shape, env.action_space.n

    sampler = Sampler(env, batch_size=1, n_steps=int(1e6))
    model = MLP(state_shape, n_actions)
    agent = A2C(model)

    train(sampler, agent)

    # trajectories = sampler.sample(agent)

    # print(trajectories[:, 0])
    # print(trajectories[0][2])
    # print(rewards)
    # print(states_tp1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
