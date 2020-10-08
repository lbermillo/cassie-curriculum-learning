import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256), init_weights=False):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim[0]), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(hidden_dim[0], hidden_dim[1]), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(hidden_dim[1], 1), )

        # Q2 architecture
        self.l4 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim[0]), nn.ReLU())
        self.l5 = nn.Sequential(nn.Linear(hidden_dim[0], hidden_dim[1]), nn.ReLU())
        self.l6 = nn.Sequential(nn.Linear(hidden_dim[1], 1), )

        if init_weights:
            self.apply(weights_init)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        # Q1 forward
        x1 = self.l1(xu)
        x1 = self.l2(x1)
        x1 = self.l3(x1)

        # Q2 forward
        x2 = self.l4(xu)
        x2 = self.l5(x2)
        x2 = self.l6(x2)

        return x1, x2
