import torch.nn as nn
import torch


class ActorCritic(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(ActorCritic, self).__init__()

        self.state_shape = state_shape
        self.n_actions = n_actions

        self.common_dense = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(),
        )

        self.value_function = nn.Linear(256, 1)
        self.logits = nn.Linear(256, n_actions)

        self.relu = nn.ReLU()

    def forward(self, states):
        out = self.common_dense(states)
        value_function = self.value_function(out)
        logits = self.logits(out)

        return logits, value_function
