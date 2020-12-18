#!/usr/bin/env python3

import time
import torch
import atexit
import pickle
import argparse
import platform
import numpy as np

from rl.networks.Actor import Actor

from cassie.envs import cassie_standing
from cassie.cassiemujoco.cassieUDP import *
from cassie.utils.quaternion_function import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *


def log(filename, times, states, inputs, outputs, targets, sto="final"):
    data = {"time": times, "state": states, "input": inputs, "output": outputs, "target": targets}

    filep = open(filename + "_log" + str(sto) + ".pkl", "wb")

    pickle.dump(data, filep)

    filep.close()


def rotate_to_orient(vec, target_orientation):
    quaternion = euler2quat(z=target_orientation[2], y=target_orientation[1], x=target_orientation[0])
    iquaternion = inverse_quaternion(quaternion)

    if len(vec) == 3:
        return rotate_by_quaternion(vec, iquaternion)

    elif len(vec) == 4:
        new_orient = quaternion_product(iquaternion, vec)
        if new_orient[0] < 0:
            new_orient = -new_orient
        return new_orient


parser = argparse.ArgumentParser(description='Cassie Sim Controller')
parser.add_argument('--filename', '-f', action='store', default=None, required=True,
                    help='Provide filename for logs (default=None)')
parser.add_argument('--load', '-l', action='store', default=None, dest='load', required=True,
                    help='Provide path to existing model to load it (default=None)')
parser.add_argument('--simrate', type=int, default=60,
                        help='Simulation rate in Hz (default: 60)')
parser.add_argument('--reduced_input', action='store_true', default=False,
                        help='Trains with inputs that are directly measured only (default: False)')
parser.add_argument('--no_clock', action='store_false', default=True, dest='clock',
                        help='Disables clock')
parser.add_argument('--hidden', type=int, nargs='+', default=(256, 256),
                        help='Size of the 2 hidden layers (default=[256, 256])')
parser.add_argument('--learn_PD', action='store_true', default=False, dest='learn_PD',
                        help='Adds PD gains to the action space. Number of actions will become 30 instead of 10')

args = parser.parse_args()

# Prepare model
env = cassie_standing.CassieEnv(simrate=args.simrate,
                                reduced_input=args.reduced_input,
                                clock_based=args.clock,
                                learn_PD=args.learn_PD,)

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# initialize action tracker
prev_action = np.zeros(action_dim)

# command consists of [forward velocity, lateral velocity, yaw rate]
command = np.zeros(3)

# load policy
checkpoint = torch.load(args.load, map_location=torch.device('cpu'))

policy = Actor(state_dim, action_dim, max_action, args.hidden)
policy.load_state_dict(checkpoint['actor'])
policy.eval()

# Initialize logging variables
time_log   = [] # time stamp
state_log  = [] # cassie_state
input_log  = [] # network inputs
output_log = [] # network outputs
target_log = [] #PD target log

# run log function when closing the application
atexit.register(log, args.filename, time_log, state_log, input_log, output_log, target_log)

# Initialize control structure with gains
P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
u = pd_in_t()

for i in range(5):
    u.leftLeg.motorPd.pGain[i] = P[i]
    u.leftLeg.motorPd.dGain[i] = D[i]
    u.rightLeg.motorPd.pGain[i] = P[i+5]
    u.rightLeg.motorPd.dGain[i] = D[i+5]

pos_index = np.array([2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

# Determine whether running in simulation or on the robot
if platform.node() == 'cassie':
    cassie = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                       local_addr='10.10.10.100', local_port='25011')
else:
    cassie = CassieUdp() # local testing

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

# Whether or not STO has been TOGGLED (i.e. it does not count the initial STO condition)
# STO = True means that STO is ON (i.e. robot is not running) and STO = False means that STO is
# OFF (i.e. robot *is* running)
sto = True
sto_count = 0

# We have multiple modes of operation
# 0: Normal operation, walking with policy
# 1: Start up, Standing Pose with variable height (no balance)
# 2: Stop Drop and hopefully not roll, Damping Mode with no P gain
operation_mode = 0

D_mult = 1

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
        command[2] -= state.radio.channel[3] / 60.0

        # Reset orientation on STO
        if state.radio.channel[8] < 0:
            command[2] = quaternion2euler(state.pelvis.orientation[:])[2]

            # initialize action tracker
            prev_action = np.zeros(action_dim)

            # Save log files after STO toggle (skipping first STO)
            if sto is False:
                log(args.filename, time_log, state_log, input_log, output_log, target_log, sto=str(sto_count))
                sto_count += 1
                sto = True
                # Clear out logs
                time_log = []  # time stamp
                state_log = []
                input_log = []  # network inputs
                output_log = []  # network outputs
                target_log = []  # PD target log
        else:
            sto = False

        # Switch the operation mode based on the toggle next to STO

        # towards operator means damping shutdown mode
        if state.radio.channel[9] < -0.5:
            operation_mode = 2

        # away from the operator TODO: assign this to walking in the future
        elif state.radio.channel[9] > 0.5:
            operation_mode = 1

        # middle is standing, makes more sense to startup Cassie in standing mode
        else:
            operation_mode = 0

    else:
        # Automatically change orientation and speed
        tt = time.monotonic() - t0
        command[2] += 0  # math.sin(t / 8) / 400

        # Keep speed at 0 for now to test standing policy
        speed = 0.

    #------------------------------- 0: Standing ---------------------------
    if operation_mode == 0:
        
        # Reassign because it might have been changed by the damping mode
        for i in range(5):
            u.leftLeg.motorPd.pGain[i] = P[i]
            u.leftLeg.motorPd.dGain[i] = D[i]
            u.rightLeg.motorPd.pGain[i] = P[i+5]
            u.rightLeg.motorPd.dGain[i] = D[i+5]

        # command consists of [forward velocity, lateral velocity, yaw rate]
        command = np.zeros(3)

        # create external state
        ext_state = np.concatenate((prev_action, command))

        if args.clock:
            # Clock is muted for standing
            clock = [0., 0.]

            # Concatenate clock with ext_state
            ext_state = np.concatenate((clock, ext_state))

        # Use state estimator
        robot_state = np.concatenate([
            # pelvis height
            [state.pelvis.position[2] - state.terrain.height],

            # pelvis orientation
            rotate_to_orient(state.pelvis.orientation[:], (0, 0, command[2])),

            # pelvis linear/translational velocity
            rotate_to_orient(state.pelvis.translationalVelocity[:], (0, 0, command[2])),

            # pelvis rotational/angular velocity
            state.pelvis.rotationalVelocity[:],

            # joint positions w/ added noise
            state.motor.position[:],  # actuated joint positions
            state.joint.position[:],  # unactuated joint positions

            # joint velocities
            state.motor.velocity[:],  # actuated joint velocities
            state.joint.velocity[:]  # unactuated joint velocities
        ])

        # Concatenate robot_state to ext_state
        RL_state = np.concatenate((robot_state, ext_state))

        # flatten state
        torch_state = torch.FloatTensor(RL_state.reshape(1, -1))

        # select action according to actor's current policy
        action = policy(torch_state).cpu().data.numpy().flatten()

        # update previous action
        prev_action = action

        if args.learn_PD:
            # map policy actions to hardware
            target_P = P + action[10:20] * min(P)
            target_D = D + action[20:]   * min(D)
            action   = action[:10]

        # apply neutral offset to action
        target = action + offset

        # Send action
        for i in range(5):
            u.leftLeg.motorPd.pGain[i] = P[i] if not args.learn_PD else target_P[i]
            u.leftLeg.motorPd.dGain[i] = D[i] if not args.learn_PD else target_D[i]

            u.rightLeg.motorPd.pGain[i] = P[i + 5] if not args.learn_PD else target_P[i + 5]
            u.rightLeg.motorPd.dGain[i] = D[i + 5] if not args.learn_PD else target_D[i + 5]

            u.leftLeg.motorPd.pTarget[i] = target[i]
            u.rightLeg.motorPd.pTarget[i] = target[i + 5]

        cassie.send_pd(u)

        # Logging
        if not sto:
            time_log.append(time.time())
            state_log.append(state)
            input_log.append(RL_state)
            output_log.append(action)
            target_log.append(target)

        # Measure delay
        print('delay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))

    #------------------------------- TODO: 1: Walking/Running ---------------------------
    # elif operation_mode == 1:

        # print('Startup Standing. Height = ' + str(standing_height))
        # #Do nothing
        # # Reassign with new multiplier on damping
        # for i in range(5):
        #     u.leftLeg.motorPd.pGain[i] = 0.0
        #     u.leftLeg.motorPd.dGain[i] = 0.0
        #     u.rightLeg.motorPd.pGain[i] = 0.0
        #     u.rightLeg.motorPd.dGain[i] = 0.0
        #
        # # Send action
        # for i in range(5):
        #     u.leftLeg.motorPd.pTarget[i] = 0.0
        #     u.rightLeg.motorPd.pTarget[i] = 0.0
        # cassie.send_pd(u)

    #------------------------------- Shutdown Damping ---------------------------
    elif operation_mode == 2:

        print('Shutdown Damping. Multiplier = ' + str(D_mult))
        # Reassign with new multiplier on damping
        for i in range(5):
            u.leftLeg.motorPd.pGain[i] = 0.0
            u.leftLeg.motorPd.dGain[i] = D_mult*D[i]
            u.rightLeg.motorPd.pGain[i] = 0.0
            u.rightLeg.motorPd.dGain[i] = D_mult*D[i+5]

        # Send action
        for i in range(5):
            u.leftLeg.motorPd.pTarget[i] = 0.0
            u.rightLeg.motorPd.pTarget[i] = 0.0
        cassie.send_pd(u)

    #---------------------------- Other, should not happen -----------------------
    else:
        print('Error, In bad operation_mode with value: ' + str(operation_mode))

    # Measure delay
    print('delay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))
