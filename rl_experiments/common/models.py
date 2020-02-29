import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, state_shape, n_actions):

        super().__init__()
        self.n_actions = n_actions
        self.state_shape = state_shape

        assert len(state_shape) == 1
        state_dim = state_shape[0]

        self.dense1 = nn.Linear(state_dim, 64)
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, n_actions)
        self.relu = nn.ReLU()

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch states, shape = [batch_size, *state_dim=4]
        """
        # Use your network to compute qvalues for given state
        out = self.relu(self.dense1(state_t))
        out = self.relu(self.dense2(out))
        qvalues = self.dense3(out)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()


class Cnn(nn.Module):
    def __init__(self, state_shape, n_actions):

        super().__init__()
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(3136, 512)
        self.dense2 = nn.Linear(512, n_actions)

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch states, shape = [batch_size, *state_dim=4]
        """
        # Use your network to compute qvalues for given state
        out = self.relu(self.conv1(state_t))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.dense1(self.flatten(out)))
        qvalues = self.dense2(out)

        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"
        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()


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
