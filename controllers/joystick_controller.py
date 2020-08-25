from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *

import time

import numpy as np
import torch
import pickle
from rl.policies import GaussianMLP
import platform
from quaternion_function import *
from math import floor

import upload_log_gdrive as gDrive

#import signal 
import atexit
import sys
import datetime

time_log   = [] # time stamp
input_log  = [] # network inputs
output_log = [] # network outputs 
state_log  = [] # cassie state
target_log = [] #PD target log

#PREFIX = "./"


PREFIX = "/home/drl/jdao/cassie-rl-testing/"
#PREFIX = "/home/robot/Desktop/Testing/jdao_cassie-rl-testing/" #Dylan's Prefix

if len(sys.argv) > 1:
    filename = PREFIX + "logs/" + sys.argv[1]
else:
    filename = PREFIX + "logs/" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M')

def log(sto="final"):
    data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target": target_log}

    filep = open(filename + "_log" + str(sto) + ".pkl", "wb")

    pickle.dump(data, filep)

    
    filep.close()
    # gDrive.CreateFolders(filep.name) #upload to google drive

atexit.register(log)

# Prevent latency issues by disabling multithreading in pytorch
torch.set_num_threads(1)

# Prepare model
phase = 0
counter = 0
phase_add = 1
speed = 0
simrate = 60
phaselen = floor(1682/simrate) - 1

# policy = torch.load(PREFIX + "trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2.pt")
# policy.bounded = False
# policy = torch.load(PREFIX + "trained_models/5k_randfric_randjoint_(1).pt")
# policy = torch.load(PREFIX + "trained_models/5k_footorient_actpenaltybig_footheightvellow_randjoint_randfric_seed10.pt")
# policy = torch.load(PREFIX + "trained_models/fwrd_walk_StateEst_tasktrajmatch_fixedtdvel_test.pt")
policy = torch.load(PREFIX + "trained_models/fwrd_walk_StateEst_speed0-3_freq1-2.pt")
policy.eval()

max_speed = 3.0
min_speed = 0.0
max_y_speed = 0.0
min_y_speed = 0.0

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
pos_mirror_index = np.array([2,3,4,5,6,21,22,23,28,29,30,34,7,8,9,14,15,16,20])
vel_mirror_index = np.array([0,1,2,3,4,5,19,20,21,25,26,27,31,6,7,8,12,13,14,18])
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

orient_add = 0

# We have multiple modes of operation
# 0: Normal operation, walking with policy
# 1: Start up, Standing Pose with variable height (no balance)
# 2: Stop Drop and hopefully not roll, Damping Mode with no P gain
operation_mode = 0
standing_height = 0.7
MAX_HEIGHT = 0.8
MIN_HEIGHT = 0.4
D_mult = 1  # Reaaaaaally bad stability problems if this is pushed higher as a multiplier
            # Might be worth tuning by joint but something else if probably needed

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

            # Save log files after STO toggle (skipping first STO)
            if sto is False:
                log(sto_count)
                sto_count += 1
                sto = True
                # Clear out logs
                time_log   = [] # time stamp
                input_log  = [] # network inputs
                output_log = [] # network outputs 
                state_log  = [] # cassie state
                target_log = [] #PD target log
        else:
            sto = False

        # Switch the operation mode based on the toggle next to STO
        if state.radio.channel[9] < -0.5: # towards operator means damping shutdown mode
            operation_mode = 2
            #D_mult = 5.5 + 4.5* state.radio.channel[7]     # Tune with right side knob 1x-10x (went unstable really fast)
                                                            # Consider using this for some sort of p gain based 

        elif state.radio.channel[9] > 0.5: # away from the operator means that standing pose
            operation_mode = 1
            standing_height = MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT)*0.5*(state.radio.channel[6] + 1)

        else:                               # Middle means normal walking 
            operation_mode = 0
        
        curr_max = 1#max_speed / 2# + (max_speed / 2)*state.radio.channel[4]
        speed_add = 0#(max_speed / 2) * state.radio.channel[4]
        speed = max(min_speed, state.radio.channel[0] * curr_max + speed_add)
        speed = min(max_speed, state.radio.channel[0] * curr_max + speed_add)
        
        # print("speed: ", speed)
        phase_add = 1+state.radio.channel[5]
        # env.y_speed = max(min_y_speed, -state.radio.channel[1] * max_y_speed)
        # env.y_speed = min(max_y_speed, -state.radio.channel[1] * max_y_speed)
    else:
        # Automatically change orientation and speed
        tt = time.monotonic() - t0
        orient_add += 0#math.sin(t / 8) / 400
        #env.speed = 0.2
        speed += 0.001#((math.sin(tt / 2)) * max_speed)
        print("speed: ", speed)
        #if env.phase % 14 == 0:
        #	env.speed = (random.randint(-1, 1)) / 2.0
        # print(env.speed)
        speed = max(min_speed, speed)
        speed = min(max_speed, speed)
        # env.y_speed = (math.sin(tt / 2)) * max_y_speed
        # env.y_speed = max(min_y_speed, env.y_speed)
        # env.y_speed = min(max_y_speed, env.y_speed)

    #------------------------------- Normal Walking ---------------------------
    if operation_mode == 0:

        # Reassign because it might have been changed by the damping mode
        for i in range(5):
            u.leftLeg.motorPd.pGain[i] = P[i]
            u.leftLeg.motorPd.dGain[i] = D[i]
            u.rightLeg.motorPd.pGain[i] = P[i+5]
            u.rightLeg.motorPd.dGain[i] = D[i+5]

        clock = [np.sin(2 * np.pi *  phase / phaselen), np.cos(2 * np.pi *  phase / phaselen)]
        # quaternion = euler2quat(z=orient_add, y=state.radio.channel[6]/10, x=state.radio.channel[7]/10)
        quaternion = euler2quat(z=orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)
        new_orient = quaternion_product(iquaternion, state.pelvis.orientation[:])
        
        # Adding bias to quat (ROLL PITCH YAW)
        # euler_orient = quaternion2euler(new_orient) + [state.radio.channel[6]/20, state.radio.channel[7]/20, 0]
        # new_orient = euler2quat(z=euler_orient[2], y=euler_orient[1], x=euler_orient[0])

        if new_orient[0] < 0:
            new_orient = -new_orient

        # new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
        new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
        new_translationalVelocity += np.array([state.radio.channel[6]/10, state.radio.channel[7]/10, 0])
        # print("new vel: ", new_translationalVelocity)
        new_translationalAcceleration = rotate_by_quaternion(state.pelvis.translationalAcceleration[:], iquaternion)
        # print('new_orientation: {}'.format(new_orient))
        
            
        ext_state = np.concatenate((clock, [speed]))
        robot_state = np.concatenate([
                [state.pelvis.position[2] - state.terrain.height], # pelvis height
                new_orient,                                     # pelvis orientation
                state.motor.position[:],                        # actuated joint positions

                new_translationalVelocity[:],                   # pelvis translational velocity
                state.pelvis.rotationalVelocity[:],             # pelvis rotational velocity 
                state.motor.velocity[:],                        # actuated joint velocities

                new_translationalAcceleration[:],
                # state.pelvis.translationalAcceleration[:],      # pelvis translational acceleration
                
                state.joint.position[:],                        # unactuated joint positions
                state.joint.velocity[:]                         # unactuated joint velocities
        ])
        RL_state = np.concatenate([robot_state, ext_state])
        
        #pretending the height is always 1.0
        # RL_state[0] = 1.0
        
        # Construct input vector
        torch_state = torch.Tensor(RL_state)
        # torch_state = shared_obs_stats.normalize(torch_state)

        # Get action
        _, action = policy.act(torch_state, True)
        env_action = action.data.numpy()
        target = env_action + offset

        # Send action
        for i in range(5):
            u.leftLeg.motorPd.pTarget[i] = target[i]
            u.rightLeg.motorPd.pTarget[i] = target[i+5]
        cassie.send_pd(u)

        # Logging
        if sto == False:
            time_log.append(time.time())
            state_log.append(state)
            input_log.append(RL_state)
            output_log.append(env_action)
            target_log.append(target)
    #------------------------------- Start Up Standing ---------------------------
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

    # Track phase
    phase += phase_add
    if phase > phaselen:
        phase = 0
        counter += 1
    # print("phase: {}\tcounter: {}\tphaselen: {}".format(phase, counter, phaselen))
