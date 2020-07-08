#!/usr/bin/env python

import time
import torch
import argparse
import numpy as np

from cassie import cassie_env
from rl.policies.Actor import Actor

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helper functions
def visualize(env, actor, max_steps, runs=5, dt=0.033, speedup=1):
    
    for run in range(runs):
        episode_reward = 0

        with torch.no_grad():
            state = env.reset()
            done  = False
            step  = 0

            while step < max_steps and not done:

                action = actor.act(state)

                state, reward, done, info = env.step(action)

                episode_reward += reward

                step += 1

                time.sleep(dt / speedup)  
                
            print("----------------------------------------")
            print("Episode Reward {}: {}".format(run, round(episode_reward, 2)))
            print("----------------------------------------")        

class TD3Actor:
    def __init__(self, state_dim, action_dim, max_action):
        
        # initialize actor network
        self.actor  = Actor(state_dim, action_dim, max_action, device=device).to(device)
        
        
    def act(self, state):
        # flatten state
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        # select action according to actor's current policy
        return self.actor(state).cpu().data.numpy().flatten()


    def load(self, path, device):
        # load checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # load actor state dict
        self.actor.load_state_dict(checkpoint['actor'])
        
        self.actor.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Policy test script for TD3')
    parser.add_argument('--env', type=int, default=0,
                        help='Cassie environment: [0] Delta, [1] Delta w/ SE, [2] Delta w/ SE no TA,[3] No Delta w/ SE no TA (default: 0)')
    parser.add_argument('--policy', action='store',
                        help='Policy to be tested')
    parser.add_argument('--steps', type=int, default=300,
                        help='Number of steps in each episode (default: 300)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of episodes to run (default: 10)')
    args = parser.parse_args()
    
    if args.env == 0:
        # create delta env
        env = cassie_env.CassieEnv("walking", clock_based=True, state_est=True, target_actions=False)
    elif args.env == 1:
        # create delta env w/ state est
        env = cassie_env.CassieEnv("walking", clock_based=False, state_est=True, target_actions=True)
    elif args.env == 2:
        env = cassie_env.CassieEnv("walking", clock_based=False, state_est=True, target_actions=False)
    elif args.env == 3:
        env = cassie_env.CassieEnv("walking", clock_based=True, state_est=True, target_actions=False)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Estimated to be -1.25 to 1.25
    action_range = (np.full_like(env.action_space, -1.25), np.full_like(env.action_space, 1.25))

    # Load checkpoint
    actor = TD3Actor(state_dim, action_dim, [float(action_range[1])])
    actor.load(args.policy, device=device)

    visualize(env, actor, args.steps, runs=args.runs)

    exit(0)
