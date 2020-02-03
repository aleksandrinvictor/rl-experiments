from torch.utils.tensorboard import SummaryWriter
from dqn import DQN
from utils import RewardTracker, ReplayBuffer, EpsilonTracker
from agents import VanillaDQNAgent

import click
import gym
import logging
import os
import numpy as np


def play_and_record(initial_state, agent, env, replay_buffer, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for i in range(n_steps):
        action = agent.sample_actions([s])[0]
        next_s, r, done, _ = env.step(action)

        replay_buffer.add(s, action, r, next_s, done)
        sum_rewards += r
        s = next_s
        if done:
            s = env.reset()

    return sum_rewards, s


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            action = agent.sample_actions([s], greedy)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


def make_env(env_name: str, seed=None):
    # CartPole is wrapped with a time limit wrapper by default
    if env_name == 'CartPole-v1':
        env = gym.make(env_name).unwrapped
    else:
        env = gym.make(env_name)
    if seed is not None:
        env.seed(seed)
    return env


def run(
    env: object,
    env_name: str,
    agent: object,
    total_steps: int,
    # reward_tracker: RewardTracker,
    eps_tracker: EpsilonTracker,
    writer: SummaryWriter,
    replay_buffer: ReplayBuffer = None,
    batch_size: int = None,
    timesteps_per_epoch: int = 1
):
    # total_steps = 4 * 10**4
    eval_freq = 1000
    state = env.reset()

    for step in np.arange(total_steps + 1):

        _, state = play_and_record(
            state,
            agent,
            env,
            replay_buffer,
            timesteps_per_epoch
        )

        agent.epsilon = eps_tracker(step)
        writer.add_scalar('epsilon', agent.epsilon, step)

        batch = replay_buffer.sample(batch_size)

        agent.update(batch, step, writer=writer)

        if step % eval_freq == 0:
            # eval the agent
            eval_reward = evaluate(
                make_env(env_name, seed=int(step)),
                agent,
                n_games=3,
                greedy=True,
                t_max=1000
            )
            logger.info(
                f'step: {step}, mean_reward_per_episode: {eval_reward}')
            writer.add_scalar('eval/mean_reward_per_episode',
                              eval_reward, step)


def fill_buffer(state, env, agent, replay_buffer):
    for i in range(100):
        play_and_record(state, agent, env, replay_buffer, n_steps=10**2)

    return replay_buffer


@click.command()
@click.option('-e', '--env_name', type=str, default='CartPole-v1')
@click.option('-t', '--total_steps', type=int, default=4*10**4)
@click.option('-target', '--target', type=float, default=None)
@click.option('-gamma', '--gamma', type=float, default=0.99)
@click.option('-start_eps', '--start_epsilon', type=float, default=1.0)
@click.option('-end_eps', '--end_epsilon', type=float, default=0.01)
@click.option('-n_iter_decay', '--n_iter_decay', type=int, default=4*10**4)
@click.option('-refresh_freq', '--refresh_target_network_freq', type=float, default=100)
@click.option('-replay_size', '--replay_buffer_size', type=int, default=10**4)
@click.option('-batch_size', '--replay_batch_size', type=int, default=32)
@click.option('-lr', '--learning_rate', type=float, default=1e-4)
@click.option('-o', '--output_path', type=str, default='./runs')
def main(
    env_name: str,
    total_steps: int,
    target: float,
    gamma: float,
    start_epsilon: float,
    end_epsilon: float,
    n_iter_decay: int,
    refresh_target_network_freq: int,
    replay_buffer_size: int,
    replay_batch_size: int,
    learning_rate: float,
    output_path: str
):

    logger.info('Starting experiment:')
    logger.info(f'Environment: {env_name}')

    hparams = {key: val for (key, val) in locals().items() if val is not None}
    writer = SummaryWriter(output_path)
    writer.add_hparams(
        hparams,
        {}
    )

    # Setup environment
    env = make_env(env_name)
    state_shape, n_actions = env.observation_space.shape, env.action_space.n
    # Init replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)
    # Init reward tracker
    # reward_tracker = RewardTracker(rewards_num, target)
    # Init epsilon tracker
    eps_tracker = EpsilonTracker(start_epsilon, end_epsilon, n_iter_decay)

    # Create newural network to approximate Q-values
    model = DQN(state_shape, n_actions)
    # Create Q-learning agent
    agent = VanillaDQNAgent(
        model,
        refresh_target_network_freq,
        start_epsilon,
        gamma,
        learning_rate
    )
    # Fill replay buffer with tuples (state, action, reward, next_state, done)
    replay_buffer = fill_buffer(env.reset(), env, agent, replay_buffer)

    # Run training
    run(
        env,
        agent,
        total_steps,
        # reward_tracker,
        eps_tracker,
        writer,
        replay_buffer,
        replay_batch_size
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
