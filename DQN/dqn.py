import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, state_shape, n_actions):

        super().__init__()
        self.n_actions = n_actions
        self.state_shape = state_shape
        # Define your network body here. Please make sure agent is fully contained here
        assert len(state_shape) == 1
        state_dim = state_shape[0]

        self.dense1 = nn.Linear(state_dim, 64)
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, n_actions)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch states, shape = [batch_size, *state_dim=4]
        """
        # Use your network to compute qvalues for given state
        out1 = self.relu1(self.dense1(state_t))
        out2 = self.relu2(self.dense2(out1))
        qvalues = self.dense3(out2)

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
