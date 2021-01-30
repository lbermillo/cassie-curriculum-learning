import time
import torch
import numpy as np

from rl.algorithms.SAC import SAC
from rl.utils.ReplayMemory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, algorithm, state_dim, action_space, max_action, hidden_dim=(256, 128), actor_lr=3e-4, critic_lr=3e-4,
                 discount=0.99, tau=5e-3, policy_noise=0.2, noise_clip=0.5, random_action_steps=1e4, use_mirror_loss=True,
                 capacity=1e6, batch_size=100, policy_update_freq=2, termination_curriculum=None, chkpt_pth=None,
                 init_weights=True, writer=None, alpha=0.2, policy_type="Gaussian", target_update_interval=1,
                 automatic_entropy_tuning=False):

        self.action_dim = action_space.shape[0]
        self.max_action = float(max_action)
        self.use_mirror_loss = use_mirror_loss
        self.random_action_steps = random_action_steps

        # initialize SAC model
        self.model = SAC(state_dim, action_space, discount, tau, alpha, policy_type, target_update_interval,
                         automatic_entropy_tuning, hidden_dim, critic_lr, actor_lr, device)

        # load existing model when provided
        if chkpt_pth is not None:

            # load model
            self.model.load(chkpt_pth, device)

            # DEBUG check if it stays relatively the same policy w/out random actions
            self.random_action_steps = 0

        # intialize replay buffer and batch size
        self.replay_buffer = ReplayMemory(capacity)
        self.batch_size = batch_size

        # initialize policy update frequency
        self.policy_update_freq = policy_update_freq

        # initialize counter to track total number of steps and updates
        self.total_steps = 0
        self.total_updates = 0

        # initialize writer for logging
        self.writer = writer

        # initialize parameters for Termination Curriculum
        self.tc = termination_curriculum

    def policy(self, state):
        # select action randomly for the given steps
        if self.total_steps < self.random_action_steps:
            return np.random.uniform(-self.max_action, self.max_action, size=self.action_dim)

        return self.model.act(state, evaluate=False)

    def sample_buffer(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)

        # convert batches to tensors
        state      = torch.FloatTensor(batch.state).to(device)
        action     = torch.FloatTensor(batch.action).to(device)
        next_state = torch.FloatTensor(batch.next_state).to(device)
        reward     = torch.FloatTensor(batch.reward).to(device).unsqueeze(1)
        done       = 1 - torch.FloatTensor(batch.done).to(device).unsqueeze(1)

        return state, action, next_state, reward, done

    def update(self, num_updates=1):
        # skip update if the replay buffer is less than the batch size
        if len(self.replay_buffer) < self.batch_size:
            return

        # start the updates
        for step in range(num_updates):
            # sample experiences from replay buffer
            state, action, next_state, reward, done = self.sample_buffer(self.batch_size)

            # Update parameters of all the networks
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.model.update_parameters(
                state, action, next_state, reward, done, updates=self.total_updates)

            self.total_updates += 1

        if self.writer:
            self.writer.add_scalar('loss/critic', critic_1_loss, self.total_steps)
            self.writer.add_scalar('loss/actor',  policy_loss,   self.total_steps)

    def collect(self, env, max_steps, num_updates=1, reset_ratio=0, use_phase=False):
        # set environment to training mode
        env.train()

        # initialize episode reward tracker for logging
        episode_reward = 0

        # reset environment
        state = env.reset(reset_ratio=reset_ratio, use_phase=use_phase)
        done = False
        step = 0

        # collect experiences and store in replay buffer
        while step < max_steps and not done:
            # get action from policy
            action = self.policy(state)

            # update model
            self.update(num_updates)

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

        # set environment to eval mode
        env.eval()

        # initialize reward tracker
        total_rewards = 0.

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
              filename=None, reset_ratio=0, use_phase=False):
        episode = 0
        best_score = 0.0

        while self.total_steps < training_steps:

            # collect experiences
            episode_steps, episode_reward = self.collect(env, max_steps, reset_ratio=reset_ratio, use_phase=use_phase)

            if self.writer:
                # log episode reward to tensorboard
                self.writer.add_scalar('reward/train', episode_reward, episode)

            # evaluate current policy
            if episode % evaluate_interval == 0 and self.total_steps > self.batch_size:

                # get evaluation score
                score = self.evaluate(env, max_steps=max_steps, reset_ratio=reset_ratio, use_phase=use_phase)

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


