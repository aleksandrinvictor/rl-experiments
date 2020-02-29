from typing import NoReturn
from .algorithm import Actor
from rl_experiments.common.sampler import Sampler
from rl_experiments.common.utils import make_env

import numpy as np


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


def train_on_policy(
    env_name: str,
    sampler: Sampler,
    agent: Actor,
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
