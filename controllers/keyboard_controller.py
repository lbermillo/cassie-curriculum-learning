from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *
# from cassie.speed_env import CassieEnv_speed
# from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
# from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
# from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot

import time
import numpy as np
import torch
import pickle
from rl.policies import GaussianMLP
import platform
from quaternion_function import *

# import signal
import atexit
import sys
import select
import tty
import termios

time_log = []  # time stamp
input_log = []  # network inputs
output_log = []  # network outputs
state_log = []  # cassie state
target_log = []  # PD target log

filename = "test.p"
filep = open(filename, "wb")

PREFIX = "/home/drl/jdao/jdao_cassie-rl-testing/"

if len(sys.argv) > 1:
    filename = PREFIX + "logs/" + sys.argv[1]
else:
    filename = PREFIX + "logs/" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M')


def log(sto="final"):
    data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target": target_log}

    filep = open(filename + "_log" + str(sto) + ".pkl", "wb")

    pickle.dump(data, filep)

    filep.close()


atexit.register(log)

max_speed = 1.5
min_speed = 0.0


def log():
    data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target": target_log}
    pickle.dump(data, filep)


def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


atexit.register(log)

# Prevent latency issues by disabling multithreading in pytorch
torch.set_num_threads(1)

# Prepare model
# env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)
# env.reset_for_test()
phase = 0
counter = 0
phase_add = 1
speed = 0
y_speed = 0

policy = torch.load("./trained_models/sidestep_StateEst_speedreward.pt")
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
pos_mirror_index = np.array([2, 3, 4, 5, 6, 21, 22, 23, 28, 29, 30, 34, 7, 8, 9, 14, 15, 16, 20])
vel_mirror_index = np.array([0, 1, 2, 3, 4, 5, 19, 20, 21, 25, 26, 27, 31, 6, 7, 8, 12, 13, 14, 18])
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
        while time.monotonic() - t < 60 / 2000:
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
            if isData():
                c = sys.stdin.read(1)
                if c == 'w':
                    speed += .1
                    print("Increasing speed to: ", speed)
                elif c == 's':
                    speed -= .1
                    print("Decreasing speed to: ", speed)
                elif c == 'a':
                    y_speed += .1
                    print("Increasing y speed to: ", y_speed)
                elif c == 'd':
                    y_speed -= .1
                    print("Decreasing y speed to: ", y_speed)
                elif c == 'j':
                    phase_add += .1
                    print("Increasing frequency to: ", phase_add)
                elif c == 'h':
                    phase_add -= .1
                    print("Decreasing frequency to: ", phase_add)
                elif c == 'l':
                    orient_add += .1
                    print("Increasing orient_add to: ", orient_add)
                elif c == 'k':
                    orient_add -= .1
                    print("Decreasing orient_add to: ", orient_add)
                elif c == 'p':
                    print("Applying force")
                    push = 100
                    push_dir = 0
                    force_arr = np.zeros(6)
                    force_arr[push_dir] = push
                    env.sim.apply_force(force_arr)
                else:
                    pass
            speed = max(min_speed, speed)
            speed = min(max_speed, speed)
            y_speed = max(min_y_speed, y_speed)
            y_speed = min(max_y_speed, y_speed)
            print("speed: ", speed)
            print("y_speed: ", y_speed)
            print("frequency: ", phase_add)

        clock = [np.sin(2 * np.pi * phase / 27), np.cos(2 * np.pi * phase / 27)]
        # euler_orient = quaternion2euler(state.pelvis.orientation[:])
        # print("euler orient: ", euler_orient + np.array([orient_add, 0, 0]))
        # new_orient = euler2quat(euler_orient + np.array([orient_add, 0, 0]))
        quaternion = euler2quat(z=orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)
        new_orient = quaternion_product(iquaternion, state.pelvis.orientation[:])
        if new_orient[0] < 0:
            new_orient = -new_orient
        new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
        # print('new_orientation: {}'.format(new_orient))

        ext_state = np.concatenate((clock, [speed, y_speed]))
        robot_state = np.concatenate([
            [state.pelvis.position[2] - state.terrain.height],  # pelvis height
            new_orient,
            # state.pelvis.orientation[:],                                 # pelvis orientation
            state.motor.position[:],  # actuated joint positions

            state.pelvis.translationalVelocity[:],  # pelvis translational velocity
            # new_translationalVelocity[:],
            state.pelvis.rotationalVelocity[:],  # pelvis rotational velocity
            state.motor.velocity[:],  # actuated joint velocities

            state.pelvis.translationalAcceleration[:],  # pelvis translational acceleration

            state.joint.position[:],  # unactuated joint positions
            state.joint.velocity[:]  # unactuated joint velocities
        ])
        RL_state = np.concatenate([robot_state, ext_state])

        # pretending the height is always 1.0
        RL_state[0] = 1.0

        # Construct input vector
        torch_state = torch.Tensor(RL_state)
        # torch_state = shared_obs_stats.normalize(torch_state)

        # Get action
        _, action = policy.act(torch_state, True)
        env_action = action.data.numpy()
        target = env_action + offset

        # print(state.pelvis.position[2] - state.terrain.height)

        # Send action
        for i in range(5):
            u.leftLeg.motorPd.pTarget[i] = target[i]
            u.rightLeg.motorPd.pTarget[i] = target[i + 5]
        # time.sleep(0.005)
        cassie.send_pd(u)

        # Measure delay
        print('delay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))

        # Logging
        time_log.append(time.time())
        state_log.append(state)
        input_log.append(RL_state)
        output_log.append(env_action)
        target_log.append(target)

        # Track phase
        phase += phase_add
        if phase >= 28:
            phase = 0
            counter += 1

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
