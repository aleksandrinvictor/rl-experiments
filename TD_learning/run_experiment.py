from torch.utils.tensorboard import SummaryWriter
from agents import QLearningAgent, EVSarsaAgent
from utils import RewardTracker, ReplayBuffer

import click
import gym
import logging
import os
import gym.envs.toy_text
import numpy as np
from pandas import DataFrame

AGENTS = {
    'qlearning': QLearningAgent,
    'evsarsa': EVSarsaAgent
}


def play_and_train(
    env,
    agent,
    t_max=10**4,
    replay_buffer: ReplayBuffer = None,
    replay_batch_size: int = None
):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        # get agent to pick action given state s.
        a = agent.get_action(s)

        next_s, r, done, _ = env.step(a)

        # train (update) agent for state s
        agent.update(s, a, r, next_s)

        if replay_buffer is not None:
            # store current <s,a,r,s'> transition in buffer
            replay_buffer.add(s, a, r, next_s, done)

            # sample replay_batch_size random transitions from replay,
            # then update agent on each of them in a loop
            s_, a_, r_, next_s_, done_ = replay_buffer.sample(
                replay_batch_size)
            for i in range(replay_batch_size):
                agent.update(s_[i], a_[i], r_[i], next_s_[i])

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


def run(
    env: object,
    agent: object,
    episodes_num: int,
    reward_tracker: RewardTracker,
    eps_decay: float,
    writer: SummaryWriter,
    replay_buffer: ReplayBuffer = None,
    replay_batch_size: int = None
):

    for i in range(episodes_num):
        reward = play_and_train(
            env,
            agent,
            replay_buffer=replay_buffer,
            replay_batch_size=replay_batch_size
        )
        reward_tracker.push(reward)
        agent.epsilon *= eps_decay

        if i % 100 == 0:
            mean_reward = reward_tracker.get_mean_reward()
            logger.info(f'episode: {i}, mean_reward: {mean_reward}')
            writer.add_scalar('mean_reward', mean_reward, i)


@click.command()
@click.option('-e', '--env_name', type=str, default='Taxi-v3')
@click.option('-a', '--agent_type', type=str, default='qlearning')
@click.option('-n', '--episodes_num', type=int, default=1000)
@click.option('-rn', '--rewards_num', type=int, default=10)
@click.option('-target', '--target', type=float, default=0)
@click.option('-alpha', '--alpha', type=float, default=0.5)
@click.option('-eps', '--epsilon', type=float, default=0.25)
@click.option('-eps_decay', '--epsilon_decay', type=float, default=0.99)
@click.option('-disc', '--discount', type=float, default=0.99)
@click.option('-replay', '--replay_buffer_size', type=int, default=None)
@click.option('-replay_batch', '--replay_batch_size', type=int, default=32)
@click.option('-o', '--output_path', type=str, default='./runs')
def main(
    env_name: str,
    agent_type: str,
    episodes_num: int,
    rewards_num: int,
    target: float,
    alpha: float,
    epsilon: float,
    epsilon_decay: float,
    discount: float,
    replay_buffer_size: int,
    replay_batch_size: int,
    output_path: str
):

    logger.info('Starting experiment')
    logger.info(f'Environment: {env_name}')
    env = gym.make(env_name)
    # env = gym.envs.toy_text.CliffWalkingEnv()
    n_actions = env.action_space.n
    # print(env.render())

    logger.info(f'Using agent of type {agent_type}')
    agent = AGENTS[agent_type](
        alpha,
        epsilon,
        discount,
        get_legal_actions=lambda s: range(n_actions)
    )

    replay_buffer = None
    if replay_buffer_size is not None:
        replay_buffer = ReplayBuffer(replay_buffer_size)

    logger.info('Init writer')

    writer = SummaryWriter(output_path)
    writer.add_hparams(
        {
            'agent_type': agent_type,
            'episodes_num': episodes_num,
            'rewards_num': rewards_num,
            'target': target,
            'alpha': alpha,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'discount': discount,
            'replay_buffer_size': '-' if replay_buffer_size is None else replay_buffer_size,
            'replay_batch_size': replay_batch_size
        },
        {}
    )

    logger.info('Init reward tracker')
    reward_tracker = RewardTracker(rewards_num, target)

    logger.info('Running experiment')
    run(
        env,
        agent,
        episodes_num,
        reward_tracker,
        epsilon_decay,
        writer,
        replay_buffer,
        replay_batch_size
    )

    logger.info('Done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
