from torch.utils.tensorboard import SummaryWriter
from rl_experiments.common.replay_buffer import ReplayBuffer
from rl_experiments.common.models import MLP, Cnn, DuelingCnn
from rl_experiments.common.utils import LinearDecay, make_env, get_env_type
from agents import VanillaDQNAgent, DoubleDQNAgent
from tqdm import tqdm, trange

import click
import gym
import logging
import os
import numpy as np
import torch
import random


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


def run(
    env: object,
    env_name: str,
    agent: object,
    total_steps: int,
    eps_tracker: EpsilonTracker,
    eval_frequency: int,
    writer: SummaryWriter,
    replay_buffer: ReplayBuffer = None,
    batch_size: int = None,
    timesteps_per_epoch: int = 1,
):

    state = env.reset()

    for step in trange(int(total_steps + 1)):

        # Play timesteps_per_epoch and return cumulative reward and last state
        _, state = play_and_record(
            state,
            agent,
            env,
            replay_buffer,
            timesteps_per_epoch
        )
        # print(evaluate(env, agent, n_games=15))

        # Update epsilon
        agent.epsilon = eps_tracker(step)
        writer.add_scalar('epsilon', agent.epsilon, step)

        # Sample batch and train agent
        batch = replay_buffer.sample(batch_size)
        agent.update(batch, step, writer=writer)

        # print(evaluate(env, agent, n_games=15))

        if step % eval_frequency == 0:
            # Eval the agent
            eval_reward = evaluate(
                make_env(env_name, seed=int(step)),
                agent,
                n_games=15,
                greedy=True,
                t_max=100
            )
            logger.info(
                f'step: {step}, mean_reward_per_episode: {eval_reward}'
            )
            writer.add_scalar(
                'eval/mean_reward_per_episode',
                eval_reward,
                step
            )


def fill_buffer(state, env, agent, replay_buffer, replay_size):
    for i in tqdm(range(100), 'Filling replay buffer'):
        play_and_record(state, agent, env, replay_buffer, n_steps=10**2)

        if len(replay_buffer) == replay_size:
            break

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
        evaluate(env_monitor, agent, n_games=1, greedy=True)
        for _ in range(10)
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
@click.option('-n_iter_decay', '--n_iter_decay', type=float, default=4*10**4)
@click.option('-refresh_freq', '--refresh_target_network_freq', type=float, default=100)
@click.option('-replay_size', '--replay_buffer_size', type=int, default=10**4)
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
    n_iter_decay: int,
    refresh_target_network_freq: int,
    replay_buffer_size: int,
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

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Init replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)
    # Init epsilon tracker
    eps_tracker = LinearDecay(start_epsilon, end_epsilon, n_iter_decay)

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
    if agent_type == 'vanilla':
        agent = VanillaDQNAgent(
            model,
            target_network,
            refresh_target_network_freq,
            start_epsilon,
            gamma,
            learning_rate,
            device
        )
    elif agent_type == 'double':
        agent = DoubleDQNAgent(
            model,
            target_network,
            refresh_target_network_freq,
            start_epsilon,
            gamma,
            learning_rate,
            device
        )
    # print(evaluate(env, agent, n_games=15))
    # Fill replay buffer with tuples (state, action, reward, next_state, done)
    replay_buffer = fill_buffer(
        env.reset(),
        env,
        agent,
        replay_buffer,
        replay_buffer_size
    )

    # Run training
    run(
        env,
        env_name,
        agent,
        total_steps,
        eps_tracker,
        eval_frequency,
        writer,
        replay_buffer,
        replay_batch_size,
    )

    # Record video with learned agent
    # record_video(env_name, agent, output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
