from numpy import ndarray
from typing import Tuple, NoReturn, List
from torch import tensor
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch
import copy
import numpy as np


class VanillaDQNAgent:

    def __init__(
        self,
        model: nn.Module,
        target_network: nn.Module,
        refresh_target_network_freq: int,
        epsilon: float,
        gamma: float = 0.99,
        lr: float = 1e-4,
        device: str = 'cpu'
    ):

        self.model = model
        # self.model.to(device)

        self.target_network = target_network
        self.refresh_target_network_freq = refresh_target_network_freq
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_grad_norm = 50
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def update(
        self,
        batch: Tuple[ndarray, ...],
        step: int,
        writer: SummaryWriter = None
    ) -> NoReturn:
        loss = self.compute_loss(batch)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        if writer is not None:
            writer.add_scalar('train_params/grad_norm', grad_norm, step)
            writer.add_scalar('train_params/td_loss',
                              loss.data.cpu().item(), step)

        if step % self.refresh_target_network_freq == 0:
            self.target_network.load_state_dict(self.model.state_dict())

    def compute_loss(
        self,
        batch: Tuple[ndarray, ...]
    ) -> tensor:

        states, actions, rewards, next_states, is_done = batch
        # Batch of states and next states of shape: [batch_size, *state_shape]
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        next_states = torch.tensor(
            next_states, device=self.device, dtype=torch.float)

        # Batches of actions, rewards and done flag of shape: [batch_size]
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

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.target_network(next_states)

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
            range(len(actions)), actions
        ]

        # compute V*(next_states) = max_a' Q_target(s', a') using predicted next q-values
        next_state_values = torch.max(predicted_next_qvalues, -1)[0]

        # Compute "target q-values" for loss
        # If current state is the last state => r(s, a) + max_a' Q_target(s', a') = r(s, a)
        # To achieve this we will apply is_not_done mask that is 0 when episode completed
        target_qvalues_for_actions = rewards + \
            self.gamma * next_state_values * is_not_done

        # mean squared error loss to minimize
        # Detaching target_qvalues_for_actions required
        # to not pass gradient through target network
        loss = torch.mean((
            predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

        return loss

    def sample_actions(self, states: List[np.ndarray], greedy: bool = False) -> np.ndarray:
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        qvalues = self.model.get_qvalues(states)
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        if greedy:
            return best_actions
        else:
            should_explore = np.random.choice(
                [0, 1],
                batch_size,
                p=[1-epsilon, epsilon]
            )
            return np.where(should_explore, random_actions, best_actions)


class DoubleDQNAgent(VanillaDQNAgent):
    def __init__(
        self,
        model: nn.Module,
        target_network: nn.Module,
        refresh_target_network_freq: int,
        epsilon: float,
        gamma: float = 0.99,
        lr: float = 1e-4,
        device: str = 'cpu'
    ):
        super().__init__(
            model,
            target_network,
            refresh_target_network_freq,
            epsilon,
            gamma,
            lr,
            device
        )
    

    def compute_loss(
        self,
        batch: Tuple[ndarray, ...]
    ) -> tensor:

        states, actions, rewards, next_states, is_done = batch
        # Batch of states and next states of shape: [batch_size, *state_shape]
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        next_states = torch.tensor(
            next_states, device=self.device, dtype=torch.float)

        # Batches of actions, rewards and done flag of shape: [batch_size]
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

        # compute q-values for all actions in next states
        # using following folmula: V*(next_states) = Q_target(s', argmax_a' Q(s', a'))
        predicted_next_qvalues = self.model(next_states)
        optimal_actions = torch.argmax(predicted_next_qvalues, -1)[0]
        next_state_values = self.target_network(
            next_states)[:, optimal_actions]

        # Compute "target q-values" for loss
        # If current state is the last state => r(s, a) + Q_target(s', argmax_a' Q(s', a')) = r(s, a)
        # To achieve this we will apply is_not_done mask that is 0 when episode completed
        target_qvalues_for_actions = rewards + \
            self.gamma * next_state_values * is_not_done

        # mean squared error loss to minimize
        # Detaching target_qvalues_for_actions required
        # to not pass gradient through target network
        loss = torch.mean((
            predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

        return loss

