from torch import tensor
from typing import NoReturn, Dict, List, Any

import torch
import torch.nn as nn
import numpy as np


class A2C:

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        entropy_coef: float = 1e-3,
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
        logits, _ = self.model(states)
        probs = nn.functional.softmax(logits, -1).detach().numpy()[0]
        return np.random.choice(len(probs), p=probs)

    def _get_return(
        self,
        rewards: np.ndarray,
    ) -> tensor:
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
        rollout: Dict[str, List[Any]]
    ) -> NoReturn:

        loss = 0

        # n - actions num
        # t - length of rollout
        # [t]
        discounted_returns = torch.tensor(
            self._get_return(rollout['rewards']),
            device=self.device,
            dtype=torch.float32
        )

        state_t = torch.tensor(
            rollout['states_t'],
            device=self.device,
            dtype=torch.float32
        )

        actions = torch.tensor(
            rollout['actions'],
            device=self.device,
        )

        logits, values = self.model(state_t)

        probs = nn.functional.softmax(logits, -1)

        log_probs = nn.functional.log_softmax(logits, -1)

        # select log-probabilities for chosen actions, log pi(a_i|s_i)
        actions_selected_mask = torch.nn.functional.one_hot(
            actions,
            num_classes=log_probs.shape[1]
        )

        log_probs_for_actions = torch.sum(
            log_probs * actions_selected_mask,
            dim=1
        )

        state_tpn = torch.tensor(
            rollout['states_tp1'][-1],
            dtype=torch.float
        )
        _, value_tpn = self.model(state_tpn)
        value_tpn = value_tpn * self.gamma ** len(discounted_returns)

        advantage = discounted_returns + value_tpn - values
        entropy = -(probs * log_probs).sum(-1).mean()

        actor_loss = (-log_probs_for_actions * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + critic_loss - self.entropy_coef * entropy

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train(
    env_name: str,
    sampler: Sampler,
    agent: object,
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
