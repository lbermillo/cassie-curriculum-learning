#!/usr/bin/env python3

import sys
import time
import torch
import pickle
import platform
import numpy as np

from rl.networks.Actor import Actor

from cassie.envs import cassie_standing
from cassie.quaternion_function import *
from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *

# Prevent latency issues by disabling multithreading in pytorch
torch.set_num_threads(1)

# Prepare model
env = cassie_standing.CassieEnv()

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# TODO: create args from terminal for policy parameters
hidden_dim = (256, 256)
checkpoint = torch.load("./results/1. Standing/Standing[RC[0.3]TW1.0]_TD3[ALR5e-05CLR8e-05HDN(256, "
                        "256)BTCH1024TAU0.001]_Training[TS50000000ES200EXP0.1SNonePHSFalseSPD1PWR100FH0.7]100MultZPos"
                        ".chkpt")

policy = Actor(state_dim, action_dim, max_action, hidden_dim)
policy.load_state_dict(checkpoint['actor'])
policy.eval()

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
simrate = 60
orient_add = 0

while True:
    # Wait until next cycle time
    while time.monotonic() - t < simrate/2000:
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

        # Reset orientation on STO
        if state.radio.channel[8] < 0:
            orient_add = quaternion2euler(state.pelvis.orientation[:])[2]
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
        orient_add += 0  # math.sin(t / 8) / 400

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
    #------------------------------- TODO: 1: Walking/Running ---------------------------
    elif operation_mode == 1:
        print('Startup Standing. Height = ' + str(standing_height))
        #Do nothing
        # Reassign with new multiplier on damping
        for i in range(5):
            u.leftLeg.motorPd.pGain[i] = 0.0
            u.leftLeg.motorPd.dGain[i] = 0.0
            u.rightLeg.motorPd.pGain[i] = 0.0
            u.rightLeg.motorPd.dGain[i] = 0.0

        # Send action
        for i in range(5):
            u.leftLeg.motorPd.pTarget[i] = 0.0
            u.rightLeg.motorPd.pTarget[i] = 0.0
        cassie.send_pd(u)

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