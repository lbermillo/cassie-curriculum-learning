import os
import torch
import torch.nn.functional as F

from torch.optim import Adam
from rl.networks.Critic import Critic as QNetwork
from rl.networks.Actor import GaussianPolicy, DeterministicPolicy
from rl.utils.SACUtils import soft_update, hard_update


class SAC(object):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy_type, target_update_interval,
                 automatic_entropy_tuning, hidden_size, critic_lr, policy_lr, device='cpu'):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy_type
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = device

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=policy_lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)

    def act(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, state_batch, action_batch, next_state_batch, reward_batch, mask_batch, updates):

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def get_state_dict(self):
        return {
            'actor': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.policy_optim.state_dict(),
            'critic_optimizer': self.critic_optim.state_dict(),
        }

    # Save model parameters
    def save(self, directory, filename):
        # create a folder in results for each run
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.get_state_dict(), '{}/{}.chkpt'.format(directory, filename))

    # Load model parameters
    def load(self, path, map_location):
        # load checkpoint
        checkpoint = torch.load(path, map_location=map_location)

        # load actor/critic state dicts
        self.policy.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        # load actor/critic traget state dicts
        self.critic_target.load_state_dict(checkpoint['critic_target'])

        # load actor/critic optimizer state dicts
        self.policy_optim.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer'])

