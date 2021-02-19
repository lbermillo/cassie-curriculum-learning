import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256), init_weights=False):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim[0]), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(hidden_dim[0], hidden_dim[1])         , nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(hidden_dim[1], 1)                     ,          )

        # Q2 architecture
        self.l4 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim[0]), nn.ReLU())
        self.l5 = nn.Sequential(nn.Linear(hidden_dim[0], hidden_dim[1])         , nn.ReLU())
        self.l6 = nn.Sequential(nn.Linear(hidden_dim[1], 1)                     ,          )

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


class LSTMCritic(Critic):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256), init_weights=False):
        super(LSTMCritic, self).__init__(state_dim, action_dim, hidden_dim, init_weights)

        # Q1 architecture
        self.lstm_q1 = nn.LSTMCell(hidden_dim[1], hidden_dim[1])
        self.cx_q1   = torch.zeros(1,             hidden_dim[1])
        self.hx_q1   = torch.zeros(1,             hidden_dim[1])

        # Q2 architecture
        self.lstm_q2 = nn.LSTMCell(hidden_dim[1], hidden_dim[1])
        self.cx_q2   = torch.zeros(1,             hidden_dim[1])
        self.hx_q2   = torch.zeros(1,             hidden_dim[1])

        self.hidden_dim = hidden_dim

    def reset_lstm_hidden_state(self, done=True):
        if done is True:
            self.cx_q1 = torch.zeros(1, self.hidden_dim[1])
            self.hx_q1 = torch.zeros(1, self.hidden_dim[1])

            self.cx_q2 = torch.zeros(1, self.hidden_dim[1])
            self.hx_q2 = torch.zeros(1, self.hidden_dim[1])
        else:
            self.cx_q1 = self.cx_q1.data
            self.hx_q1 = self.hx_q1.data

            self.cx_q2 = self.cx_q2.data
            self.hx_q2 = self.hx_q2.data

    def forward(self, state, action, hidden_states=None):
        xu = torch.cat([state, action], 1)

        # Q1 forward
        x1 = self.l1(xu)
        x1 = self.l2(x1)

        if hidden_states is None:
            hx_q1, cx_q1 = self.lstm_q1(x1, (self.hx_q1, self.cx_q1))
            self.hx_q1 = hx_q1
            self.cx_q1 = cx_q1
        else:
            hx_q1, cx_q1 = self.lstm_q1(x1, hidden_states)

        x1 = self.l3(hx_q1)

        # Q2 forward
        x2 = self.l4(xu)
        x2 = self.l5(x2)

        if hidden_states is None:
            hx_q2, cx_q2 = self.lstm_q2(x2, (self.hx_q2, self.cx_q2))
            self.hx_q2 = hx_q2
            self.cx_q2 = cx_q2
        else:
            hx_q2, cx_q2 = self.lstm_q2(x2, hidden_states)

        x2 = self.l6(hx_q2)

        return x1, x2

