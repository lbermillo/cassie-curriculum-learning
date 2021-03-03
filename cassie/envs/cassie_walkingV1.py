import os
import gym
import random
import numpy as np

from math import floor
from cassie.utils.quaternion_function import *
from cassie.trajectory import CassieTrajectory
from cassie.rewards.cost import power_cost
from cassie.rewards.reward import compute_reward
from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis


class CassieEnv:
    def __init__(self, speed=(0, 1), simrate=50, reward_cutoff=0.3, debug=False, config="cassie/cassiemujoco/cassie.xml"):

        # Using CassieSim
        self.config = config
        self.sim = CassieSim(self.config)
        self.vis = None

        # Initialize parameters
        self.simrate = simrate
        self.reward_cutoff = reward_cutoff
        self.debug = debug
        self.config = config
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
        self.speed = np.array(speed) * 10.

        # action offset so that the policy can learn faster and prevent standing on heels
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968,
                                0.0045, 0.0, 0.4973, -1.1997, -1.5968])

        self.P = np.array([100., 100., 88., 96., 50.])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()
        self.cassie_state = state_out_t()

        # Command input will consist of XY Velocity and Yaw Rate (Z Angular Velocity)
        self.cmd_input = np.zeros(3)

        # total number of phases
        self.trajectory = CassieTrajectory(os.path.join(os.path.dirname(__file__), "..", "trajectory", "stepdata.bin"))
        self.phaselen   = floor(len(self.trajectory) / self.simrate) - 1

        # Initialize Observation and Action Spaces
        self.observation_space = np.zeros(len(self.get_full_state()))
        self.action_space = gym.spaces.Box(-1.25 * np.ones(10), 1.25 * np.ones(10), dtype=np.float32)

    def step_simulation(self, action):
        # Create Target Action
        target = action + self.offset

        self.u = pd_in_t()
        # Apply Action
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i] = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i] = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i] = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        # Send action input (u) into sim and update cassie_state
        self.cassie_state = self.sim.step_pd(self.u)

    def step(self, action):

        # simulating delays
        simrate = self.simrate + np.random.randint(-5, 5)

        # reset mujoco tracking variables
        foot_pos = np.zeros(6)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)

        for _ in range(simrate):
            self.step_simulation(action)

            # Relative Foot Position tracking
            self.sim.foot_pos(foot_pos)
            self.l_foot_pos += foot_pos[0:3]
            self.r_foot_pos += foot_pos[3:6]

        # average foot positions
        self.l_foot_pos /= self.simrate
        self.r_foot_pos /= self.simrate
        foot_pos = np.append(self.l_foot_pos, self.r_foot_pos)

        # Pelvis position and velocity
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        # Calculate total reward
        total_reward = compute_reward(self.cmd_input, self.cassie_state, qpos, qvel, foot_pos, rw=(0, 0, 1, 0, 0), debug=self.debug) \
                       - 0.125 * power_cost(self.cassie_state, np.sum(self.sim.get_body_mass()), qvel, debug=self.debug)

        # Early termination conditions
        done = not (0.4 < qpos[2] < 3.0) or total_reward < self.reward_cutoff

        # return state, reward, and done flag
        return self.get_full_state(), total_reward, done, {'fall': (0.4 < qpos[2] < 3.0)}

    def reset(self):
        # reset robot state
        self.sim.set_const()

        # get the corresponding state from the reference trajectory for the current phase
        init_qpos, init_qvel = self.get_ref_state(random.randint(0, self.phaselen), np.random.randint(self.speed[1]) / 10.)

        # set initial joint positions and velocities
        self.sim.set_qpos(init_qpos)
        self.sim.set_qvel(init_qvel)

        # get updated robot state
        self.cassie_state = self.sim.step_pd(self.u)

        # set new speed command
        if self.speed[0] != self.speed[1]:
            self.cmd_input[0] = np.random.randint(self.speed[0], self.speed[1])
        else:
            self.cmd_input[0] = self.speed[0]
        self.cmd_input[0] /= 10.

        # return initial state
        return self.get_full_state()

    def get_full_state(self):
        robot_state = np.concatenate([

            # Pelvis States
            rotate_to_orient(self.cassie_state.pelvis.orientation[:]),
            self.cassie_state.pelvis.rotationalVelocity[:],

            # Motor States
            self.cassie_state.motor.position[:],
            self.cassie_state.motor.velocity[:],

        ])

        # Concatenate robot_state to ext_state
        ext_state = np.concatenate((robot_state, self.cmd_input))

        return ext_state

    def get_ref_state(self, phase, speed):
        if phase > self.phaselen:
            phase = 0

        pos    = np.copy(self.trajectory.qpos[phase * self.simrate])
        pos[0] = pos[0] * speed + ((self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * speed)
        pos[1] = 0

        vel = np.copy(self.trajectory.qvel[phase * self.simrate])
        vel[0] *= speed

        return pos, vel

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, self.config)

        self.vis.draw(self.sim)
