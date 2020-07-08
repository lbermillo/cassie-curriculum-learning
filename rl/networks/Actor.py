import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_layer=(256, 256), init_weights=False):
        super(Actor, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(state_dim, hidden_layer[0]),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_layer[0], hidden_layer[1]),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_layer[1], action_dim),
            nn.Tanh()
        )

        if init_weights:
            self.apply(weights_init)

        self.max_action = max_action

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x) * self.max_action

        return x
