from torch import tensor
from typing import NoReturn

import torch
import torch.nn as nn
import numpy as np


class A2C:

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        entropy_coef: float = 1e-2,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):

        self.model = model.to(device)
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.device = device

    def get_actions(self, states: np.ndarray):
        states = torch.tensor(
            states,
            device=self.device,
            dtype=torch.float
        )
        logits = self.model(states)
        probs = nn.functional.softmax(logits, -1).detach().numpy()[0]
        return np.random.choice(len(probs), p=probs)

    def _get_returns(
        self,
        rewards: np.ndarray,  # rewards at each step
    ) -> tensor:

        Gs = [rewards[-1]]
        for r in rewards[-2::-1]:
            Gs.append(r + self.gamma * Gs[-1])

        return Gs[::-1]

    def update(
        self,
        trajectories: np.ndarray
    ) -> NoReturn:

        loss = 0

        for trajectory in trajectories:

            # t - length of trajectory
            # n - actions number
            # state_shape - shape of state space

            # [t, state_shape]
            states_t = torch.tensor(
                trajectory[0],
                device=self.device,
                dtype=torch.float32
            )
            # [t]
            actions = torch.tensor(
                trajectory[1],
                device=self.device
            )
            # [t]
            returns = torch.tensor(
                self._get_returns(trajectory[2]),
                device=self.device,
                dtype=torch.float32
            )

            # predict logits, probas and log-probas using an agent.
            # [t, n]
            logits = self.model(states_t)
            # [t, n]
            probs = nn.functional.softmax(logits, -1)
            # [t, n]
            log_probs = nn.functional.log_softmax(logits, -1)

            # select log-probabilities for chosen actions, log pi(a_i|s_i)
            # [t, n]
            actions_selected_mask = torch.nn.functional.one_hot(
                actions,
                num_classes=logits.shape[1]
            )
            # [t]
            log_probs_for_actions = torch.sum(
                log_probs * actions_selected_mask,
                dim=1
            )

            loss_trajectory = torch.sum(log_probs_for_actions * returns)
            entropy = -(probs * log_probs).sum(-1).mean()

            loss += -loss_trajectory - self.entropy_coef * entropy

        loss = loss / len(trajectories)

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
