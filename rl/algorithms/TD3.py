# Implementation based on Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
import os
from copy import deepcopy

import torch
import torch.nn as nn
from rl.networks.Actor import Actor
from rl.networks.Critic import Critic


class TD3:
    def __init__(self, state_dim, action_dim, max_action, hidden_dim, actor_lr, critic_lr,
                 discount, tau, policy_noise, noise_clip, device='cpu', init_weights=False):

        # randomly initialize critic and actor networks
        self.actor  = Actor(state_dim, action_dim, max_action, hidden_dim, init_weights=init_weights).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim, init_weights=init_weights).to(device)

        # copy actor and critic to initialize target networks
        self.actor_target  = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        # initialize optimizers for actor and critic
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # initialize loss function
        self.loss_fn = nn.MSELoss()

        # initialize max action
        self.max_action = max_action

        # initialize hyperparameters
        self.discount = discount
        self.tau = tau

        # initialize noise standard deviation and clipping parameter
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip

        # device to store and do computations on (gpu or cpu)
        self.device = device

    def act(self, state):
        # flatten state
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # select action according to actor's current policy
        return self.actor(state).cpu().data.numpy().flatten()

    def compute_critic_values(self, state, action, next_state, reward, done):
        # create clipped noise for target policy smoothing regularization
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

        # get next_action with added the created noise using target actor's policy and clamp action
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # compute target Q value using Q1 and Q2. Refer to TD3 paper section 4.2
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = reward + (done * self.discount * torch.min(target_Q1, target_Q2)).detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        return current_Q1, current_Q2, target_Q

    def update_critic(self, current_Q1, current_Q2, target_Q):
        # compute critic loss
        critic_loss = self.loss_fn(current_Q1, target_Q) + self.loss_fn(current_Q2, target_Q)

        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # return critic loss for logging
        return critic_loss

    def update_actor(self, state, reg_coeff=1e-4):
        # compute q using policy
        q1, _ = self.critic(state, self.actor(state))

        # compute actor loss
        actor_loss = -q1.mean() # + reg_coeff * torch.norm(self.actor(state)) ** 2

        # optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # return actor loss for logging
        return actor_loss

    def update_target_networks(self):
        # update critic target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # update actor target network
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def save(self, directory, filename):
        # create a folder in results for each run
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.get_state_dict(), '{}/{}.chkpt'.format(directory, filename))

    def load(self, path, map_location):
        # load checkpoint
        checkpoint = torch.load(path, map_location=map_location)

        # load actor/critic state dicts
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        # load actor/critic traget state dicts
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

        # load actor/critic optimizer state dicts
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
