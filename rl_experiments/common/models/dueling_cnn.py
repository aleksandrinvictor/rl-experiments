import torch.nn as nn
import torch


class DuelingCnn(nn.Module):
    def __init__(self, state_shape, n_actions):

        super().__init__()
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.flatten = nn.Flatten()

        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.state_value_dense = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_dense = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, state_t):
        conv_out = self.flatten(self.conv(state_t))
        state_value = self.state_value_dense(conv_out)

        advantage = self.advantage_dense(conv_out)

        qvalues = (
            state_value
            + advantage
            - 1 / self.n_actions * torch.sum(advantage, dim=1)[:, None]
        )

        return qvalues
    
    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()
