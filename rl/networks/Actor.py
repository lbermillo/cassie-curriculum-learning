import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

# From ETH Zurich paper [https://arxiv.org/pdf/1901.08652.pdf]
# Non-linearity has a strong effect on performance on the physical system and unbounded activation functions
# such as ReLU  can degrade performance on the real robot. Bounded activation functions such as Tanh, yield less
# aggressive trajectories when subject to disturbance


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=(256, 256), init_weights=False):
        super(Actor, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_dim[1], action_dim),
            nn.Tanh()
        )

        if init_weights:
            self.apply(weights_init_)

        self.max_action = max_action

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x) * self.max_action

        return x


class LSTMActor(Actor):
    def __init__(self, state_dim, action_dim, max_action=1, hidden_dim=(256, 256), init_weights=False):
        super(LSTMActor, self).__init__(state_dim, action_dim, max_action, hidden_dim, init_weights)

        self.lstm = nn.LSTMCell(hidden_dim[1], hidden_dim[1])
        self.cx = torch.zeros(1, hidden_dim[1]).to(device)
        self.hx = torch.zeros(1, hidden_dim[1]).to(device)
        self.hidden_dim = hidden_dim

    def reset_lstm_hidden_state(self, done=True):
        if done is True:
            self.cx = torch.zeros(1, self.hidden_dim[1]).to(device)
            self.hx = torch.zeros(1, self.hidden_dim[1]).to(device)
        else:
            self.cx = self.cx.data
            self.hx = self.hx.data

    def forward(self, x, hidden_states=None):
        x = self.l1(x)
        x = self.l2(x)

        if hidden_states is None:
            hx, cx = self.lstm(x, (self.hx, self.cx))
            self.hx = hx
            self.cx = cx
        else:
            # hidden_states are provided during updates
            hx, cx = self.lstm(x, hidden_states)

        x = self.l3(hx) * self.max_action

        return x, (hx, cx)


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim[0]),
            nn.Tanh()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.Tanh()
        )
        self.mean_linear = nn.Sequential(
            nn.Linear(hidden_dim[1], num_actions),
        )
        self.log_std_linear = nn.Sequential(
            nn.Linear(hidden_dim[1], num_actions),
        )

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, x):

        x = self.l1(x)
        x = self.l2(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mean = nn.Linear(hidden_dim[1], num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

