from numpy import ndarray
from typing import Tuple, NoReturn, List
from torch import tensor
from rl_experiments.common.sampler import Sampler
from rl_experiments.common.replay_buffer import ReplayBuffer
from rl_experiments.common.utils import LinearDecay, make_env
from torch.utils.tensorboard import SummaryWriter
from gym import Env
from tqdm import trange

import torch.nn as nn
import torch
import copy
import numpy as np
import copy
import logging
import os


class Agent:
    def __init__(
        self,
        algorithm_type: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        device: str
    ):

        self.algorithm_type = algorithm_type
        self.model = model
        self.target_network = copy.deepcopy(model)
        self.gamma = gamma
        self.device = device

        self.optimizer = optimizer

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        is_done: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        loss = self.compute_loss(
            states,
            actions,
            rewards,
            next_states,
            is_done
        )
        loss_val = loss.data.cpu().item()

        self.optimizer.zero_grad()
        loss.backward()

        total_grad_norm = 0
        for p in self.model.parameters():
            param_grad_norm = p.grad.data.norm(2)
            total_grad_norm += param_grad_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)

        self.optimizer.step()

        return loss_val, total_grad_norm

    def refresh_target_network(self) -> NoReturn:
        self.target_network.load_state_dict(self.model.state_dict())

    def compute_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        is_done: np.ndarray
    ) -> tensor:

        # Batch of states and next states: [batch_size, *state_shape]
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        next_states = torch.tensor(
            next_states, device=self.device, dtype=torch.float)

        # Batches of actions, rewards and done flag: [batch_size]
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        is_done = torch.tensor(
            is_done.astype('float32'),
            device=self.device,
            dtype=torch.float
        )
        is_not_done = 1 - is_done

        # get q-values for all actions in current states
        predicted_qvalues = self.model(states)

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
            range(len(actions)), actions
        ]

        if self.algorithm_type == 'vanilla':
            # compute q-values for all actions in next states
            predicted_next_qvalues = self.target_network(next_states)

            # compute V*(next_states) = max_a' Q_target(s', a') using predicted next q-values
            next_state_values = torch.max(predicted_next_qvalues, -1)[0]
        elif self.algorithm_type == 'double':
            # compute q-values for all actions in next states
            # using following folmula: V*(next_states) = Q_target(s', argmax_a' Q(s', a'))
            predicted_next_qvalues = self.model(next_states)
            optimal_actions = torch.argmax(predicted_next_qvalues, -1)[0]
            next_state_values = self.target_network(next_states)[:, optimal_actions]
        else:
            raise ValueError(self.algorithm_type)

        # Compute "target q-values" for loss
        # If current state is the last state => r(s, a) + max_a' Q_target(s', a') = r(s, a)
        # To achieve this we will apply is_not_done mask that is 0 when episode completed
        target_qvalues_for_actions = (
            rewards + self.gamma * next_state_values * is_not_done)

        loss = torch.nn.functional.smooth_l1_loss(
            predicted_qvalues_for_actions,
            target_qvalues_for_actions.detach())

        return loss

    def get_actions(self, states: List[np.ndarray], epsilon: float = 0) -> np.ndarray:
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        qvalues = self.model.get_qvalues(states)
        # epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        if epsilon == 0:  # greedy
            return best_actions[0]
        else:
            should_explore = np.random.choice(
                [0, 1],
                batch_size,
                p=[1-epsilon, epsilon]
            )
            return np.where(should_explore, random_actions, best_actions)[0]


def train(
    env: Env,
    agent: Agent,
    total_steps: int,
    rollout_n_steps: int,
    eval_frequency: int,
    writer: SummaryWriter,
    update_frequency: int = 4,
    start_epsilon: float = 1.0,
    end_epsilon: float = 0.01,
    eps_iters: int = 40000,
    refresh_target_network_freq: int = 100,
    replay_buffer_size: int = 10000,
    replay_buffer_start_size: int = 1000,
    replay_batch_size: int = 32
) -> NoReturn:

    # Get logger
    logger = logging.getLogger('__main__')

    # Init sampler
    sampler = Sampler(env, rollout_n_steps)

    # Init replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    replay_buffer = fill_buffer(
        sampler,
        agent,
        replay_buffer,
        replay_buffer_start_size)

    # Init epsilon tracker
    eps_tracker = LinearDecay(
        start_epsilon,
        end_epsilon,
        eps_iters)

    
    # Rewards for last 100 episodes
    episodes_played = 0
    episode_reward = 0
    last_100_ep_rewards = np.zeros(100)

    num_updates = 0

    for step in trange(int(total_steps + 1)):

        current_epsilon = eps_tracker(step)
        # Play one step in environment
        rollout = sampler.sample(agent, **{'epsilon': current_epsilon})
        replay_buffer.add(
            rollout['states_t'][0],
            rollout['actions'][0],
            rollout['rewards'][0],
            rollout['states_tp1'][0],
            rollout['dones'][0])
        
        episode_reward += rollout['rewards'][0]
        if rollout['dones'][0]:
            episodes_played += 1
            last_100_ep_rewards[episodes_played % 100] = episode_reward
            episode_reward = 0

        # Sample batch and train agent
        if step % update_frequency == 0:
            batch = replay_buffer.sample(replay_batch_size)
            states, actions, rewards, next_states, is_done = batch

            loss_val, grad_norm = agent.update(
                states,
                actions,
                rewards,
                next_states,
                is_done)
            
            num_updates += 1

        # Update target network
        if step % refresh_target_network_freq == 0:
            agent.refresh_target_network()

        # Log intermediate values
        writer.add_scalar(
            'train_params/loss', loss_val, step)

        writer.add_scalar(
            'train_params/grad_norm', grad_norm, step)

        writer.add_scalar('train_params/epsilon', current_epsilon, step)

        writer.add_scalar('eval/mean_100_train_reward',
                          np.mean(last_100_ep_rewards), step)

        if step % eval_frequency == 0:
            # Eval the agent
            eval_reward = evaluate(
                make_env(env.unwrapped.spec.id, seed=int(step)),
                agent,
                10000,
                **{'epsilon': 0.05})

            logger.info(
                f'step: {step}, mean_reward_per_episode: {eval_reward}')

            writer.add_scalar(
                'eval/mean_reward_per_episode',
                eval_reward,
                step)


def evaluate(env, agent, t_max=10000, **kwargs):
    """ Plays n_games full games. Returns mean reward. """
    s = env.reset()
    total_reward = 0
    n_episodes = 0
    episode_reward = 0
    for _ in range(t_max):
        action = agent.get_actions(np.array([s]), **kwargs)
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


def fill_buffer(
    sampler: Sampler,
    agent: object,
    replay_buffer: ReplayBuffer,
    replay_buffer_start_size: int,
) -> ReplayBuffer:

    for i in range(replay_buffer_start_size):
        rollout = sampler.sample(agent, **{'epsilon': 1.0})
        replay_buffer.add(
            rollout['states_t'][0],
            rollout['actions'][0],
            rollout['rewards'][0],
            rollout['states_tp1'][0],
            rollout['dones'][0]
        )

    return replay_buffer
