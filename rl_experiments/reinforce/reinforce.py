from torch import tensor
from typing import NoReturn

import torch
import torch.nn as nn
import numpy as np


class REINFORCE:

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 1e-2,
        device: str = 'cpu'
    ):

        self.model = model.to(device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.device = device

    def get_cumulative_rewards(
        self,
        rewards: np.ndarray,  # rewards at each step
    ) -> np.ndarray:
        """
        take a list of immediate rewards r(s,a) for the whole session 
        compute cumulative returns (a.k.a. G(s,a) in Sutton '16)
        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

        The simple way to compute cumulative rewards is to iterate from last to first time tick
        and compute G_t = r_t + gamma*G_{t+1} recurrently

        You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
        """
        Gs = []
        Gs.append(rewards[-1])
        n = len(rewards)
        gammas = np.array([self.gamma ** i for i in range(n)])
        for i in range(1, n):
            Gs.append(rewards[n - i - 1] + self.gamma*Gs[-1])
        Gs.reverse()
        return Gs

    def to_one_hot(
        self,
        y_tensor: tensor,
        ndims: int
    ):
        """ helper: take an integer vector and convert it to 1-hot matrix. """
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        y_one_hot = torch.zeros(
            y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
        return y_one_hot

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray
    ) -> NoReturn:

        # cast everything into torch tensors
        states = torch.tensor(
            states,
            device=self.device,
            dtype=torch.float32
        )
        actions = torch.tensor(
            actions,
            device=self.device,
            dtype=torch.int32
        )
        cumulative_returns = np.array(
            self.get_cumulative_rewards(rewards)
        )
        cumulative_returns = torch.tensor(
            cumulative_returns,
            device=self.device,
            dtype=torch.float32
        )

        # predict logits, probas and log-probas using an agent.
        logits = self.model(states)
        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)

        # select log-probabilities for chosen actions, log pi(a_i|s_i)
        actions_num = 2
        log_probs_for_actions = torch.sum(
            log_probs * self.to_one_hot(actions, actions_num),
            dim=1
        )

        # Compute loss here. Don't forgen entropy regularization with `entropy_coef`
        entropy = -(probs * log_probs).sum(-1).mean()
        loss = -torch.mean(
            log_probs_for_actions * cumulative_returns
        ) - self.entropy_coef * entropy

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
