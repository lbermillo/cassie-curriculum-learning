import time

import numpy as np
import torch

from rl.algorithms.TD3 import TD3
from rl.utils.ReplayMemory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, algorithm, state_dim, action_dim, max_action, hidden_dim=(256, 256), actor_lr=3e-4, critic_lr=3e-4,
                 discount=0.99, tau=5e-3, policy_noise=0.2, noise_clip=0.5, random_action_steps=1e4,
                 capacity=1e6, batch_size=100, policy_update_freq=2, termination_curriculum=None, chkpt_pth=None,
                 init_weights=True, writer=None):

        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.random_action_steps = random_action_steps

        if algorithm.lower() == 'td3':
            # initialize TD3 model
            self.model = TD3(state_dim, action_dim, max_action, hidden_dim, actor_lr, critic_lr,
                             discount, tau, policy_noise, noise_clip, device=device, init_weights=init_weights)

        # load existing model when provided
        if chkpt_pth is not None:

            # load model
            self.model.load(chkpt_pth, device)

            # DEBUG check if it stays relatively the same policy w/out random actions
            # self.random_action_steps = 0

        # intialize replay buffer and batch size
        self.replay_buffer = ReplayMemory(capacity)
        self.batch_size = batch_size

        # initialize policy update frequency
        self.policy_update_freq = policy_update_freq

        # initialize counter to track total number of steps
        self.total_steps = 0

        # initialize writer for logging
        self.writer = writer

        # initialize parameters for Termination Curriculum
        self.tc = termination_curriculum

    def policy(self, state, expl_noise=0.1):
        # select action randomly for the given steps
        if self.total_steps < self.random_action_steps:
            return np.random.uniform(-self.max_action, self.max_action, size=self.action_dim)

        # get action from model's current policy
        action = self.model.act(state)

        # add noise to action
        action += np.random.normal(0, self.max_action * expl_noise, size=action.shape[0])

        # returned clipped action
        return np.clip(action, -self.max_action, self.max_action)

    def sample_buffer(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)

        # convert batches to tensors
        state      = torch.FloatTensor(batch.state).to(device)
        action     = torch.FloatTensor(batch.action).to(device)
        next_state = torch.FloatTensor(batch.next_state).to(device)
        reward     = torch.FloatTensor(batch.reward).to(device).unsqueeze(1)
        done       = 1 - torch.FloatTensor(batch.done).to(device).unsqueeze(1)

        return state, action, next_state, reward, done

    def update(self, steps):
        # skip update if the replay buffer is less than the batch size
        if len(self.replay_buffer) < self.batch_size:
            return

        # start the updates
        for step in range(steps):

            # sample experiences from replay buffer
            state, action, next_state, reward, done = self.sample_buffer(self.batch_size)

            # compute target and current Q values
            current_Q1, current_Q2, target_Q = self.model.compute_critic_values(state, action, next_state, reward,
                                                                                done)

            # update critic by minimizing the loss
            critic_loss = self.model.update_critic(current_Q1, current_Q2, target_Q)

            if self.writer:
                # log episode reward to tensorboard
                self.writer.add_scalar('loss/critic', critic_loss, self.total_steps)

            # delay updates for actor and target networks
            if step % self.policy_update_freq == 0:
                # update actor policy using the sampled policy gradient
                actor_loss = self.model.update_actor(state)

                if self.writer:
                    # log episode reward to tensorboard
                    self.writer.add_scalar('loss/actor', actor_loss, self.total_steps)

                # update target networks
                self.model.update_target_networks()

    def collect(self, env, max_steps, noise=0.1, reset_ratio=0, use_phase=False):
        # initialize episode reward tracker for logging
        episode_reward = 0

        # reset environment
        state = env.reset(reset_ratio=reset_ratio, use_phase=use_phase)
        done = False
        step = 0

        # collect experiences and store in replay buffer
        while step < max_steps and not done:
            # get action from policy
            action = self.policy(state, expl_noise=noise)

            # execute action
            next_state, reward, done, _ = env.step(action)

            # store transition to replay buffer
            self.replay_buffer.push(state, action, next_state, reward, done)

            # update state
            state = next_state

            # update episode reward
            episode_reward += reward

            # increment step
            step += 1

            # increment total number of steps
            self.total_steps += 1

        return step, episode_reward

    def evaluate(self, env, eval_eps=10, max_steps=100, render=False, dt=0.033, speedup=1, print_stats=False,
                 reset_ratio=0, use_phase=False):
        total_rewards = 0.

        # TODO: in TSCL, find the worse reward from this eval and let the agent train on these inputs (speed, phase,
        #  orientation, pelvis height, etc.) longer

        for eps in range(eval_eps):
            with torch.no_grad():
                episode_reward = 0
                state = env.reset(reset_ratio=reset_ratio, use_phase=use_phase)
                done = False
                step = 0

                while step < max_steps and not done:
                    action = self.model.act(state)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    step += 1

                    if render:
                        env.render()
                        time.sleep(dt / speedup)

                # Update total rewards
                total_rewards += episode_reward

                if print_stats:
                    print('Episode {}: {:10.3f}'.format(eps, episode_reward))

        return total_rewards / eval_eps

    def train(self, env, training_steps, max_steps, evaluate_interval, expl_noise=0.1, directory='results',
              filename=None, reset_ratio=0, use_phase=False, adaptive_discount=False):
        episode = 0
        best_score = 0.0
        if adaptive_discount:
            self.model.discount = 0.25
            discount_rate = (0.99 - self.model.discount) / (0.75 * (training_steps - self.batch_size))

        while self.total_steps < training_steps:
            # collect experiences
            episode_steps, episode_reward = self.collect(env, max_steps, noise=expl_noise, reset_ratio=reset_ratio, use_phase=use_phase)

            if self.writer:
                # log episode reward to tensorboard
                self.writer.add_scalar('reward/train', episode_reward, episode)

            # update all networks after an episode
            self.update(episode_steps)

            # evaluate current policy
            if episode % evaluate_interval == 0 and self.total_steps > self.batch_size:
                score = self.evaluate(env, render=False)

                # update reward termination
                if self.tc is not None:
                    env.reward_cutoff = max(self.tc[0] - (self.total_steps / (2 * training_steps)), self.tc[1])

                if self.writer:
                    # log eval rewards to tensorboard
                    self.writer.add_scalar('reward/test', score, episode)

                # only save the highest reward
                if score > best_score:
                    # update best score
                    best_score = score

                    if filename:
                        # save the model state dictionary for inference
                        self.model.save(directory, filename)

            episode += 1

            if adaptive_discount and self.total_steps > self.random_action_steps:
                # update discount factor for the next episode
                self.model.discount = 0.25 + (0.99 * self.total_steps) / (training_steps - self.batch_size) \
                    if self.model.discount < 0.99 else 0.99

                print(self.total_steps, self.model.discount)



