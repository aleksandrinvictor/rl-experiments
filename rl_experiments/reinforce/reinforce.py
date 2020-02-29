from torch import tensor
from typing import NoReturn, List, Any

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

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        """ Takes actions for given states
        Args:
            states: states from trajectory
        Returns:
            np.ndarray of choosen actions.
            len(returned actions) = len(states)
        """
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
        rewards: np.ndarray,
    ) -> List[float]:
        """ Count discounted returns from rewards at each step
        Args:
            rewards: rewards at each step
        Returns:
            Array of discounted returns for each timestep
        """
        Gs = [rewards[-1]]
        for r in rewards[-2::-1]:
            Gs.append(r + self.gamma * Gs[-1])

        return Gs[::-1]

    def update(
        self,
<<<<<<< HEAD
        trajectory: np.ndarray
    ) -> NoReturn:
        """Complete gradient step.
=======
        trajectories: np.ndarray
    ) -> NoReturn:
        """Complete gradient step.
        Approximate gradient counted using the following formula:
        grad J = 1 / N * sum_N sum_T grad log_pi_theta(a_t | s_t) * (sum_t'=t^T gamma^(t' - t) * r(s_t, a_t))
        where:
            t - length of trajectory
            N - number of trajectories
>>>>>>> 815a478e1747be6598008207ccbaa2b9404ce7e7
        Args:
            trajectories: batch of trajectories (states_t, actions, rewards, states_tp1)
        """

<<<<<<< HEAD
        # t - length of trajectory
        # n - actions number
        # state_shape - shape of state space

        # [t, state_shape]
        states_t = torch.tensor(
            trajectory['states_t'],
            device=self.device,
            dtype=torch.float32
        )
        # [t]
        actions = torch.tensor(
            trajectory['actions'],
            device=self.device
        )
        # [t]
        returns = torch.tensor(
            self._get_returns(trajectory['rewards']),
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

        loss_trajectory = torch.mean(log_probs_for_actions * returns)
        entropy = -(probs * log_probs).sum(-1).mean()

        loss = -loss_trajectory - self.entropy_coef * entropy
=======
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
>>>>>>> 815a478e1747be6598008207ccbaa2b9404ce7e7

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
