from reinforce import REINFORCE
from torch.utils.tensorboard import SummaryWriter
from rl_experiments.common.models import MLP, Cnn
from rl_experiments.common.utils import make_env, get_env_type

import gym
import numpy as np
import torch
import torch.nn as nn
import click
import logging
import os
import random


def predict_probs(agent: REINFORCE, states: np.ndarray):
    """ 
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :returns: numpy array of shape [batch, n_actions]
    """
    # convert states, compute logits, use softmax to get probability
    states = torch.tensor(states, dtype=torch.float)
    logits = agent.model(states)
    return nn.functional.softmax(logits).detach().numpy()


def generate_session(env, agent, t_max=1000):
    """ 
    play a full session with REINFORCE agent and train at the session end.
    returns sequences of states, actions andrewards
    """
    # arrays to record session
    states, actions, rewards = [], [], []
    s = env.reset()

    for t in range(t_max):
        # action probabilities array aka pi(a|s)
        action_probs = predict_probs(agent, np.array([s]))[0]

        # Sample action with given probabilities.
        a = np.random.choice([0, 1], p=action_probs)
        new_s, r, done, info = env.step(a)

        # record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break

    return states, actions, rewards


def evaluate(env, agent, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    s = env.reset()
    total_reward = 0
    n_episodes = 0
    episode_reward = 0
    for _ in range(t_max):
        action_probs = predict_probs(agent, np.array([s]))[0]
        # Sample action with given probabilities.
        action = np.random.choice([0, 1], p=action_probs)
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


def record_video(env_name: str, agent: object, output_path: str):
    env_monitor = gym.wrappers.Monitor(
        make_env(env_name),
        directory=os.path.join(output_path, "videos"),
        force=True
    )
    sessions = [
        evaluate(env_monitor, agent, t_max=10000)
    ]
    env_monitor.close()


def train(
    env: object,
    env_name: str,
    agent: object,
    # total_steps: int,
    # eval_frequency: int,
    # writer: SummaryWriter,
    # batch_size: int = None,
    # timesteps_per_epoch: int = 1,
):
    for i in range(100):
        for j in range(100):
            states, actions, rewards = generate_session(env, agent)
            agent.update(states, actions, rewards)

        # Eval the agent
        eval_reward = evaluate(
            make_env(env_name, seed=int(i)),
            agent,
            t_max=10000
        )
        logger.info(
            f'step: {i}, mean_reward_per_episode: {eval_reward}'
        )

    return None


@click.command()
@click.option('-e', '--env_name', type=str, default='CartPole-v1')
@click.option('-t', '--total_steps', type=float, default=4*10**4)
@click.option('-gamma', '--gamma', type=float, default=0.99)
@click.option('-lr', '--learning_rate', type=float, default=1e-4)
@click.option('-entropy', '--entropy_coef', type=float, default=0.1)
@click.option('-eval', '--eval_frequency', type=int, default=1000)
@click.option('-o', '--output_path', type=str, default='./runs')
@click.option('-seed', '--seed', type=int, default=42)
def main(
    env_name: str,
    total_steps: int,
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

    env_type = get_env_type(env_name)
    # Create neural network to predict actions probs pi(a|s)
    if env_type == 'atari':
        model = Cnn(state_shape, n_actions).to(device)
    elif env_type == 'classic_control':
        model = MLP(state_shape, n_actions).to(device)

    agent = REINFORCE(
        model,
        learning_rate,
        gamma,
        entropy_coef,
        device
    )

    train(
        env,
        env_name,
        agent
    )

    record_video(env_name, agent, output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
