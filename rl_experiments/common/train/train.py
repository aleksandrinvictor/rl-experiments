from typing import NoReturn, Callable
from .algorithm import Actor
from rl_experiments.common.sampler import Sampler
from rl_experiments.common.utils import make_env
from rl_experiments.common.replay_buffer import ReplayBuffer
from tqdm import trange

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


def evaluate_off_policy(env, agent, greedy=False, t_max=10000):
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


def train_off_policy(
    env_name: str,
    sampler: Sampler,
    agent: object,
    total_steps: int,
    eval_frequency: int,
    eps_tracker: Callable,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    prioritized_replay: bool = False,
    prioritized_beta_tracker: Callable = None,
    prioritized_replay_eps: float = None
) -> NoReturn:

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
        # writer.add_scalar('epsilon', agent.epsilon, step)

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
            writer=None
        )

        if prioritized_replay:
            # Update priorities in replay buffer
            new_priorities = np.abs(td_errors) + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)

        if step % eval_frequency == 0:
            # Eval the agent
            eval_reward = evaluate_off_policy(
                make_env(env_name, seed=int(step)),
                agent,
                greedy=True,
                t_max=10000
            )
            print(
                f'step: {step}, mean_reward_per_episode: {eval_reward}'
            )
            # writer.add_scalar(
            #     'eval/mean_reward_per_episode',
            #     eval_reward,
            #     step
            # )


def fill_buffer(
    sampler: Sampler,
    agent: object,
    replay_buffer: ReplayBuffer,
    replay_buffer_start_size: int,
) -> ReplayBuffer:

    for i in range(replay_buffer_start_size):
        rollout = sampler.sample(agent)
        replay_buffer.add(
            rollout['states_t'][0],
            rollout['actions'][0],
            rollout['rewards'][0],
            rollout['states_tp1'][0],
            rollout['dones'][0]
        )

    return replay_buffer
