#!/usr/bin/env python3

import time
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from rl.networks.Actor import Actor
from rl.utils.ReplayMemory import ReplayMemory

from cassie.envs import cassie_standing
from cassie.utils.quaternion_function import *
from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *


def plot_motor_data(motor1, motor2, label1, label2, title=None, xlabel=None, ylabel=None):
    plt.plot(motor1, label=label1)
    plt.plot(motor2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser(description='Cassie Curriculum Learning')

parser.add_argument('--file', '-f', action='store', default=None, required=True,
                    help='Provide pkl file to read')
parser.add_argument('--load', '-l', action='store', default=None, dest='load', required=True,
                    help='Provide path to existing model to load it (default=None)')
parser.add_argument('--simrate', type=int, default=60,
                    help='Simulation rate in Hz (default: 60)')


args = parser.parse_args()

if __name__ == '__main__':
    infile = open(args.file, 'rb')
    log_data = pickle.load(infile)
    infile.close()

    l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_toe = [], [], [], [], []
    r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_toe = [], [], [], [], []

    # l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_toe, \
    #     r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_toe = np.array(log_data['output']).T

    # for state in log_data['state']:
    #     l_hip_roll.append(state.motor.velocity[0])
    #     l_hip_yaw.append(state.motor.velocity[1])
    #     l_hip_pitch.append(state.motor.velocity[2])
    #     l_knee.append(state.motor.velocity[3])
    #     l_toe.append(state.motor.velocity[4])
    #
    #     r_hip_roll.append(state.motor.velocity[5])
    #     r_hip_yaw.append(state.motor.velocity[6])
    #     r_hip_pitch.append(state.motor.velocity[7])
    #     r_knee.append(state.motor.velocity[8])
    #     r_toe.append(state.motor.velocity[9])

    r_buffer = ReplayMemory(1e4)
    hardware_state  = log_data['input']
    hardware_action = log_data['output']

    # Prepare model
    env = cassie_standing.CassieEnv(simrate=args.simrate, debug=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # load policy
    checkpoint = torch.load(args.load)

    policy = Actor(state_dim, action_dim, max_action)
    policy.load_state_dict(checkpoint['actor'])
    policy.eval()

    episode_reward = 0
    steps = 0
    sim2real_diff = []

    # flatten state
    state = torch.FloatTensor(np.array(hardware_state[0]).reshape(1, -1))

    for step in range(len(hardware_state) - 2):
        # flatten state
        state = torch.FloatTensor(np.array(hardware_state[step]).reshape(1, -1))

        # select action according to actor's current policy
        action = policy(state).cpu().data.numpy().flatten()

        # execute action
        sim_state, reward, done, _ = env.step(action)
        # _, reward, done, _ = env.step(action)

        sim_state = torch.FloatTensor(np.array(sim_state).reshape(1, -1))

        sim2real_diff.append(np.linalg.norm(state - sim_state))

        print('[{}] Action: '.format(step), action)

        # l_hip_roll.append(env.cassie_state.motor.torque[0])
        # l_hip_yaw.append(env.cassie_state.motor.torque[1])
        # l_hip_pitch.append(env.cassie_state.motor.torque[2])
        # l_knee.append(env.cassie_state.motor.torque[3])
        # l_toe.append(env.cassie_state.motor.torque[4])
        #
        # r_hip_roll.append(env.cassie_state.motor.torque[5])
        # r_hip_yaw.append(env.cassie_state.motor.torque[6])
        # r_hip_pitch.append(env.cassie_state.motor.torque[7])
        # r_knee.append(env.cassie_state.motor.torque[8])
        # r_toe.append(env.cassie_state.motor.torque[9])

        l_hip_roll.append(action[0])
        l_hip_yaw.append(action[1])
        l_hip_pitch.append(action[2])
        l_knee.append(action[3])
        l_toe.append( action[4])

        r_hip_roll.append( action[5])
        r_hip_yaw.append(  action[6])
        r_hip_pitch.append(action[7])
        r_knee.append(     action[8])
        r_toe.append(      action[9])

        # add to replay buffer
        r_buffer.push(hardware_state[step], action, hardware_state[step + 1], reward, done)

        episode_reward += reward
        steps += 1

    print('Reward: {:.3f}, Step: {}'.format(episode_reward, steps))
    plt.plot(sim2real_diff)
    plt.show()

    # plot_motor_data(l_hip_roll, r_hip_roll, 'l_hip_roll', 'r_hip_roll', 'Simulation Output', 'Time', 'Action Output')
    # plot_motor_data(l_hip_yaw, r_hip_yaw, 'l_hip_yaw', 'r_hip_yaw', 'Simulation Output', 'Time', 'Action Output')
    # plot_motor_data(l_hip_pitch, r_hip_pitch, 'l_hip_pitch', 'r_hip_pitch', 'Simulation Output', 'Time', 'Action Output')
    # plot_motor_data(l_knee, r_knee, 'l_knee', 'r_knee', 'Simulation Output', 'Time', 'Action Output')
    # plot_motor_data(l_toe, r_toe, 'l_toe', 'r_toe', 'Simulation Output', 'Time', 'Action Output')
