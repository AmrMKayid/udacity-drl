import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(1, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(1, out_channels=32, kernel_size=8, stride=2)
        self.fc = nn.Linear(256, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
