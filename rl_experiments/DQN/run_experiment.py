from torch.utils.tensorboard import SummaryWriter
from rl_experiments.common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from rl_experiments.common.models import MLP, Cnn, DuelingCnn
from rl_experiments.common.utils import LinearDecay, make_env, get_env_type
from rl_experiments.common.sampler import Sampler
from dqn import DQN
from tqdm import tqdm, trange
from typing import Callable

import click
import gym
import logging
import os
import numpy as np
import torch
import random


def evaluate(env, agent, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    s = env.reset()
    total_reward = 0
    n_episodes = 0
    episode_reward = 0
    for _ in range(t_max):
        action = agent.get_actions([s], greedy)[0]
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


def train(
    sampler: Sampler,
    env_name: str,
    agent: object,
    total_steps: int,
    eps_tracker: Callable,
    eval_frequency: int,
    writer: SummaryWriter,
    replay_buffer: ReplayBuffer = None,
    prioritized_replay: bool = False,
    prioritized_beta_tracker: Callable = None,
    prioritized_replay_eps: float = None,
    batch_size: int = None
):

    for step in trange(int(total_steps + 1)):

        # Play one step in environment
        rollout = sampler.sample(agent)
        replay_buffer.add(
            rollout['states_t'][0],
            rollout['actions'][0],
            rollout['rewards'][0],
            rollout['states_tp1'][0],
            rollout['dones'][0]
        )

        # Update epsilon
        agent.epsilon = eps_tracker(step)
        writer.add_scalar('epsilon', agent.epsilon, step)

        # Sample batch and train agent
        if prioritized_replay:
            batch = replay_buffer.sample(
                batch_size,
                prioritized_beta_tracker(step)
            )
            states, actions, rewards, next_states, is_done, weights, batch_idxes = batch
        else:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, is_done = batch
            weights, batch_idxes = np.ones(len(rewards)), None

        td_errors = agent.update(
            states,
            actions,
            rewards,
            next_states,
            is_done,
            weights,
            batch_idxes,
            step,
            writer=writer
        )

        if prioritized_replay:
            # Update priorities in replay buffer
            new_priorities = np.abs(td_errors) + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)

        if step % eval_frequency == 0:
            # Eval the agent
            eval_reward = evaluate(
                make_env(env_name, seed=int(step)),
                agent,
                greedy=True,
                t_max=10000
            )
            logger.info(
                f'step: {step}, mean_reward_per_episode: {eval_reward}'
            )
            writer.add_scalar(
                'eval/mean_reward_per_episode',
                eval_reward,
                step
            )


def fill_buffer(state, env, agent, replay_buffer, replay_buffer_start_size, sampler: Sampler):
    for i in range(replay_buffer_start_size):
        rollout = sampler.sample(agent)
        replay_buffer.add(
            rollout['states_t'][0],
            rollout['actions'][0],
            rollout['rewards'][0],
            rollout['states_tp1'][0],
            rollout['dones'][0]
        )
    # play_and_record(state, agent, env, replay_buffer,
    #                 n_steps=replay_buffer_start_size)

    return replay_buffer


def record_video(env_name: str, agent: object, output_path: str):
    env_monitor = gym.wrappers.Monitor(
        make_env(env_name),
        directory=os.path.join(output_path, "videos"),
        force=True
    )
    # env = make_env(env_name)
    # env.monitor.start(os.path.join(output_path, "videos"), force=True)
    sessions = [
        evaluate(env_monitor, agent, greedy=True, t_max=100000)
    ]
    env_monitor.close()


@click.command()
@click.option('-e', '--env_name', type=str, default='CartPole-v1')
@click.option('-a', '--agent_type', type=str, default='vanilla')
@click.option('-net', '--network_type', type=str, default='dqn')
@click.option('-t', '--total_steps', type=float, default=4*10**4)
@click.option('-gamma', '--gamma', type=float, default=0.99)
@click.option('-start_eps', '--start_epsilon', type=float, default=1.0)
@click.option('-end_eps', '--end_epsilon', type=float, default=0.01)
@click.option('-eps_iters', '--eps_iters', type=float, default=4*10**4)
@click.option('-refresh_freq', '--refresh_target_network_freq', type=float, default=100)
@click.option('-replay_start_size', '--replay_buffer_start_size', type=int, default=50000)
@click.option('-replay_size', '--replay_buffer_size', type=int, default=10**4)
@click.option('-prioritized', '--prioritized_replay', type=bool, default=False)
@click.option('-prioritized_alpha', '--prioritized_replay_alpha', type=float, default=0.6)
@click.option('-prioritized_beta_start', '--prioritized_replay_beta_start', type=float, default=0.4)
@click.option('-prioritized_beta_end', '--prioritized_replay_beta_end', type=float, default=1.0)
@click.option('-prioritized_beta_iters', '--prioritized_replay_beta_iters', type=float, default=4*10**4)
@click.option('-prioritized_eps', '--prioritized_replay_eps', type=float, default=1e-6)
@click.option('-batch_size', '--replay_batch_size', type=int, default=32)
@click.option('-lr', '--learning_rate', type=float, default=1e-4)
@click.option('-max_grad', '--max_grad_norm', type=float, default=50)
@click.option('-eval', '--eval_frequency', type=int, default=1000)
@click.option('-o', '--output_path', type=str, default='./runs')
@click.option('-seed', '--seed', type=int, default=42)
def main(
    env_name: str,
    agent_type: str,
    network_type: str,
    total_steps: int,
    gamma: float,
    start_epsilon: float,
    end_epsilon: float,
    eps_iters: int,
    refresh_target_network_freq: int,
    replay_buffer_start_size: int,
    replay_buffer_size: int,
    prioritized_replay: bool,
    prioritized_replay_alpha: float,
    prioritized_replay_beta_start: float,
    prioritized_replay_beta_end: float,
    prioritized_replay_beta_iters: int,
    prioritized_replay_eps: float,
    replay_batch_size: int,
    learning_rate: float,
    max_grad_norm: float,
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

    sampler = Sampler(env, n_steps=1)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Init replay buffer
    prioritized_beta_tracker = None
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size,
            prioritized_replay_alpha
        )
        prioritized_beta_tracker = LinearDecay(
            prioritized_replay_beta_start,
            prioritized_replay_beta_end,
            prioritized_replay_beta_iters
        )
    else:
        replay_buffer = ReplayBuffer(replay_buffer_size)

    # Init epsilon tracker
    eps_tracker = LinearDecay(start_epsilon, end_epsilon, eps_iters)

    env_type = get_env_type(env_name)
    # Create neural network to approximate Q-values
    if env_type == 'atari':
        if network_type == 'dqn':
            model = Cnn(state_shape, n_actions).to(device)
            target_network = Cnn(state_shape, n_actions).to(device)
            target_network.load_state_dict(model.state_dict())
        elif network_type == 'dueling':
            model = DuelingCnn(state_shape, n_actions).to(device)
            target_network = DuelingCnn(state_shape, n_actions).to(device)
            target_network.load_state_dict(model.state_dict())

    elif env_type == 'classic_control':
        model = MLP(state_shape, n_actions).to(device)
        target_network = MLP(state_shape, n_actions).to(device)
        target_network.load_state_dict(model.state_dict())

    # Create Q-learning agent
    agent = DQN(
        agent_type,
        model,
        target_network,
        refresh_target_network_freq,
        start_epsilon,
        gamma,
        learning_rate,
        max_grad_norm,
        device
    )
    # Fill replay buffer with tuples (state, action, reward, next_state, done)
    replay_buffer = fill_buffer(
        env.reset(),
        env,
        agent,
        replay_buffer,
        replay_buffer_start_size,
        sampler
    )

    # Run training
    train(
        sampler,
        env_name,
        agent,
        total_steps,
        eps_tracker,
        eval_frequency,
        writer,
        replay_buffer,
        prioritized_replay,
        prioritized_beta_tracker,
        prioritized_replay_eps,
        replay_batch_size,
    )

    # Record video with learned agent
    record_video(env_name, agent, output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
