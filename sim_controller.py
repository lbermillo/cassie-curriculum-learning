#!/usr/bin/env python3

import time
import torch
import pickle
import argparse
import platform
import numpy as np

# import signal
import atexit
import sys
import select
import tty
import termios

from rl.networks.Actor import Actor

from cassie.envs import cassie_standing
from cassie.quaternion_function import *
from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *


def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


parser = argparse.ArgumentParser(description='Cassie Sim Controller')
parser.add_argument('--load', '-l', action='store', default=None, dest='load', required=True,
                    help='Provide path to existing model to load it (default=None)')
parser.add_argument('--simrate', type=int, default=40,
                        help='Simulation rate in Hz (default: 40)')
parser.add_argument('--hidden', type=float, nargs='+', default=(256, 256),
                        help='Size of the 2 hidden layers (default=[256, 256])')

args = parser.parse_args()


# Prevent latency issues by disabling multithreading in pytorch
torch.set_num_threads(1)

# Prepare model
env = cassie_standing.CassieEnv()

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# Load policy
checkpoint = torch.load(args.load)

policy = Actor(state_dim, action_dim, max_action, args.hidden)
policy.load_state_dict(checkpoint['actor'])
policy.eval()

max_speed = 1.0
min_speed = 0.0
max_y_speed = 0.5
min_y_speed = -0.5
symmetry = True

# Initialize control structure with gains
P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
u = pd_in_t()
for i in range(5):
    u.leftLeg.motorPd.pGain[i] = P[i]
    u.leftLeg.motorPd.dGain[i] = D[i]
    u.rightLeg.motorPd.pGain[i] = P[i + 5]
    u.rightLeg.motorPd.dGain[i] = D[i + 5]

pos_index = np.array([2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34])
vel_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31])
offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

# Determine whether running in simulation or on the robot
if platform.node() == 'cassie':
    cassie = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                       local_addr='10.10.10.100', local_port='25011')
else:
    cassie = CassieUdp()  # local testing

# Connect to the simulator or robot
print('Connecting...')
y = None
while y is None:
    cassie.send_pd(pd_in_t())
    time.sleep(0.001)
    y = cassie.recv_newest_pd()
received_data = True
print('Connected!\n')

# Record time
t = time.monotonic()
t0 = t

orient_add = 0
old_settings = termios.tcgetattr(sys.stdin)
try:
    tty.setcbreak(sys.stdin.fileno())
    while True:
        # Wait until next cycle time
        while time.monotonic() - t < args.simrate / 2000:
            time.sleep(0.001)
        t = time.monotonic()
        tt = time.monotonic() - t0

        # Get newest state
        state = cassie.recv_newest_pd()

        if state is None:
            print('Missed a cycle')
            continue

        if platform.node() == 'cassie':
            # Radio control
            orient_add -= state.radio.channel[3] / 60.0
            speed = max(min_speed, state.radio.channel[0] * max_speed)
            speed = min(max_speed, state.radio.channel[0] * max_speed)
            phase_add = state.radio.channel[5] + 1
            # env.y_speed = max(min_y_speed, -state.radio.channel[1] * max_y_speed)
            # env.y_speed = min(max_y_speed, -state.radio.channel[1] * max_y_speed)
        else:
            # Automatically change orientation and speed
            tt = time.monotonic() - t0
            orient_add += 0  # math.sin(t / 8) / 400

            # Keep speed at 0 for now to test standing policy
            speed = 0.

            if isData():
                c = sys.stdin.read(1)
                if c == 'x':
                    force = 2000
                    direction = 0
                    force_arr = np.zeros(6)
                    force_arr[int(direction)] = int(force)
                    env.sim.apply_force(force_arr)

                    print('Applied {} N in +X direction'.format(force))

                elif c == 'y':
                    force = 200
                    direction = 1
                    force_arr = np.zeros(6)
                    force_arr[int(direction)] = int(force)
                    env.sim.apply_force(force_arr)

                    print('Applied {} N in +Y direction'.format(force))

                elif c == 'z':
                    force = 200
                    direction = 2
                    force_arr = np.zeros(6)
                    force_arr[direction] = int(force)
                    env.sim.apply_force(force_arr)

                    print('Applied {} N in +Z direction'.format(force))

                elif c == 'X':
                    force = -200
                    direction = 0
                    force_arr = np.zeros(6)
                    force_arr[int(direction)] = int(force)
                    env.sim.apply_force(force_arr)

                    print('Applied {} N in -X direction'.format(force))

                elif c == 'Y':
                    force = -200
                    direction = 1
                    force_arr = np.zeros(6)
                    force_arr[int(direction)] = int(force)
                    env.sim.apply_force(force_arr)

                    print('Applied {} N in -Y direction'.format(force))

                elif c == 'Z':
                    force = -200
                    direction = 2
                    force_arr = np.zeros(6)
                    force_arr[direction] = int(force)
                    env.sim.apply_force(force_arr)

                    print('Applied {} N in -Z direction'.format(force))

                else:
                    pass

        ext_state = [speed]

        # Clock is muted for standing
        clock = [0., 0.]

        # Concatenate clock with ext_state
        ext_state = np.concatenate((clock, ext_state))

        # Use state estimator
        robot_state = np.concatenate([
            [state.pelvis.position[2] - state.terrain.height],  # pelvis height
            state.pelvis.orientation[:],  # pelvis orientation
            state.motor.position[:],  # actuated joint positions

            state.pelvis.translationalVelocity[:],  # pelvis translational velocity
            state.pelvis.rotationalVelocity[:],  # pelvis rotational velocity
            state.motor.velocity[:],  # actuated joint velocities

            state.pelvis.translationalAcceleration[:],  # pelvis translational acceleration

            state.joint.position[:],  # unactuated joint positions
            state.joint.velocity[:]  # unactuated joint velocities
        ])

        # Concatenate robot_state to ext_state
        RL_state = np.concatenate((robot_state, ext_state))

        # flatten state
        torch_state = torch.FloatTensor(RL_state.reshape(1, -1))

        # select action according to actor's current policy
        action = policy(torch_state).cpu().data.numpy().flatten()
        target = action + offset

        # Send action
        for i in range(5):
            u.leftLeg.motorPd.pTarget[i] = target[i]
            u.rightLeg.motorPd.pTarget[i] = target[i + 5]

        cassie.send_pd(u)

        # Measure delay
        print('delay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
