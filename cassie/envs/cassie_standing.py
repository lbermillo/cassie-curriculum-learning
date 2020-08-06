import os
import random
from math import floor

import gym
import numpy as np
from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from cassie.trajectory import CassieTrajectory


# Creating the Standing Environment
class CassieEnv:

    def __init__(self, simrate=60, clock_based=True, state_est=True,
                 reward_cutoff=0.3, target_action_weight=1.0, target_height=0.9, forces=(0, 0, 0), use_phase=False,
                 min_height=0.4, max_height=3.0, config="cassie/cassiemujoco/cassie.xml", traj='walking'):

        # Using CassieSim
        self.config = config
        self.sim = CassieSim(self.config)
        self.vis = None

        # Initialize parameters
        self.clock_based = clock_based
        self.state_est = state_est
        self.reward_cutoff = reward_cutoff
        self.forces = forces
        self.min_height = min_height
        self.max_height = max_height
        self.target_height = target_height

        # Cassie properties
        self.mass = np.sum(self.sim.get_body_mass())
        self.weight = self.mass * 9.81

        # L/R midfoot offset (https://github.com/agilityrobotics/agility-cassie-doc/wiki/Toe-Model)
        # self.midfoot_offset = np.array([0.1762, 0.05219, 0., 0.1762, -0.05219, 0.])
        self.midfoot_offset = np.array([0.15, 0.05219, 0., 0.15, -0.05219, 0.])

        # action offset so that the policy can learn faster and prevent standing on heels
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968,
                                0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.offset_weight = target_action_weight

        # Initialize Observation and Action Spaces (+1 is for speed input)
        self.observation_size = 40 + 1

        if self.clock_based:
            # Add two more inputs for right and left clocks
            self.observation_size += 2

        if self.state_est:
            self.observation_size += 6

        self.observation_space = np.zeros(self.observation_size)
        self.action_space = gym.spaces.Box(-1. * np.ones(10), 1. * np.ones(10))

        # Initial Actions
        self.P = np.array([100, 100, 88, 96, 50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.cassie_state = state_out_t()
        self.simrate = simrate
        self.speed = 0
        self.use_phase = use_phase

        if self.use_phase:
            dirname = os.path.dirname(__file__)
            if traj == "walking":
                traj_path = os.path.join(dirname, "..", "trajectory", "stepdata.bin")

            elif traj == "stepping":
                # traj_path = os.path.join(dirname, "trajectory", "spline_stepping_traj.pkl")
                traj_path = os.path.join(dirname, "..", "trajectory", "more-poses-trial.bin")

            self.trajectory = CassieTrajectory(traj_path)

            self.phase = 0  # portion of the phase the robot is in
            self.phaselen = floor(len(self.trajectory) / self.simrate) - 1

        # See include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

    @property
    def dt(self):
        return 1 / 2000 * self.simrate

    def close(self):
        if self.vis is not None:
            del self.vis
            self.vis = None

    def step_simulation(self, action):
        # Create Target Action
        target = action + (self.offset_weight * self.offset)

        self.u = pd_in_t()

        # Apply perturbations to the pelvis
        self.sim.apply_force([np.random.uniform(-self.forces[0], self.forces[0]),
                              np.random.uniform(-self.forces[1], self.forces[1]),
                              np.random.uniform(-self.forces[2], self.forces[2]),
                              0,
                              0])

        # Apply Action
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i] = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]

            self.u.leftLeg.motorPd.dGain[i] = self.D[i]
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

        for _ in range(self.simrate):
            self.step_simulation(action)

        # CoM Position and Velocity
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        # Foot Positions
        foot_pos = np.zeros(6)
        self.sim.foot_pos(foot_pos)

        # Foot Forces
        foot_grf = self.sim.get_foot_forces()

        # Current State
        state = self.get_full_state()

        # Early termination condition
        height_in_bounds = self.min_height < self.sim.qpos()[2] < self.max_height

        # Current Reward
        reward = self.compute_reward(qpos, qvel, foot_pos, foot_grf) \
                 - self.compute_cost(qpos, foot_pos, foot_grf) if height_in_bounds else 0.0

        # Done Condition
        done = True if not height_in_bounds or reward < self.reward_cutoff else False

        return state, reward, done, {}

    def reset(self, phase=None):
        if self.use_phase:
            self.phase = int(phase) if phase is not None else random.randint(0, self.phaselen)

            # get the corresponding state from the reference trajectory for the current phase
            qpos, qvel = self.get_ref_state(self.phase)

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(qvel)
        else:
            self.sim.full_reset()
            self.reset_cassie_state()

        return self.get_full_state()

    def reset_cassie_state(self):
        # Only reset parts of cassie_state that is used in get_full_state
        self.cassie_state.pelvis.position[:] = [0, 0, 1.01]
        self.cassie_state.pelvis.orientation[:] = [1, 0, 0, 0]
        self.cassie_state.pelvis.rotationalVelocity[:] = np.zeros(3)
        self.cassie_state.pelvis.translationalVelocity[:] = np.zeros(3)
        self.cassie_state.pelvis.translationalAcceleration[:] = np.zeros(3)
        self.cassie_state.terrain.height = 0
        self.cassie_state.motor.position[:] = [0.0045, 0, 0.4973, -1.1997, -1.5968, 0.0045, 0, 0.4973, -1.1997, -1.5968]
        self.cassie_state.motor.velocity[:] = np.zeros(10)
        self.cassie_state.joint.position[:] = [0, 1.4267, -1.5968, 0, 1.4267, -1.5968]
        self.cassie_state.joint.velocity[:] = np.zeros(6)

    def compute_reward(self, qpos, qvel, foot_pos, foot_grf, grf_tolerance=25, rw=(0.15, 0.15, 0.15, 0.2, 0.2, 0.15, 0),
                       multiplier=500):

        left_foot_pos = foot_pos[:3]
        right_foot_pos = foot_pos[3:]

        # midfoot position
        foot_pos = np.concatenate([left_foot_pos - self.midfoot_offset[:3], right_foot_pos - self.midfoot_offset[3:]])

        # A. Task Rewards

        # 1. Pelvis Orientation [https://math.stackexchange.com/questions/90081/quaternion-distance]
        target_pose = np.array([1, 0, 0, 0])
        pose_error = 1 - np.inner(qpos[3:7], target_pose) ** 2

        r_pose = np.exp(-1000 * pose_error ** 2)

        # 2. CoM Position Modulation

        # TODO: Is this still needed? Seemed to work on some policies but clashes with other rewards
        # 2a. Horizontal Position Component (target position is the center of the support polygon)
        xy_target_pos = np.array([0.5 * (foot_pos[0] + foot_pos[3]),
                                  0.5 * (foot_pos[1] + foot_pos[4])])
        xy_com_pos = np.exp(-np.sum(qpos[:2] - xy_target_pos) ** 2)

        # 2b. Vertical Position Component (robot should stand upright and maintain a certain height)
        z_target_pos = self.target_height
        z_com_pos = 1. if z_target_pos - 0.1 < qpos[2] < z_target_pos + 0.1 \
            else np.exp(-10 * (qpos[2] - z_target_pos) ** 2)

        r_com_pos = 0.5 * xy_com_pos + 0.5 * z_com_pos

        # 3. CoM Velocity Modulation
        r_com_vel = np.exp(-np.linalg.norm(qvel[:3], 1) ** 2)

        # 4. Foot Placement
        # 4a. Foot Alignment
        r_feet_align = np.exp(-multiplier * (foot_pos[0] - foot_pos[3]) ** 2)

        # 4b. Feet Width
        target_width = 0.16
        feet_width = np.linalg.norm([foot_pos[1], foot_pos[4]], 1)
        r_foot_width = 1. if target_width - 0.01 < feet_width < target_width + 0.03 \
            else np.exp(-multiplier * (feet_width - target_width) ** 2)

        # 4c. Foot Height
        r_foot_height = np.exp(-multiplier * np.linalg.norm([foot_pos[2], foot_pos[5]]) ** 2)

        # 4d. Foot Velocity
        r_foot_vel = np.exp(-np.linalg.norm([qvel[12], qvel[19]]) ** 2)

        r_foot_placement = 0.3 * r_feet_align + 0.3 * r_foot_width + 0.3 * r_foot_height + 0.1 * r_foot_vel

        # 5. Foot/Pelvis Orientation
        foot_yaw = np.array([qpos[8], qpos[22]])
        left_foot_orient = np.exp(-multiplier * (foot_yaw[0] - qpos[6]) ** 2)
        right_foot_orient = np.exp(-multiplier * (foot_yaw[1] - qpos[6]) ** 2)

        r_fp_orient = 0.5 * left_foot_orient + 0.5 * right_foot_orient

        # TODO: 6.Leg Symmetry Reward
        # r_symmetry = np.exp(-(qpos[7:21] - qpos[21:]) ** 2)

        # 7. Ground Force Modulation (Even Vertical Foot Force Distribution)
        # TODO: How to incentivize to step rather than drag?
        # GRF target discourages shear forces and incites even vertical foot force distribution
        target_grf = (foot_grf[2] + foot_grf[5]) / 2.
        left_grf = np.exp(-(np.linalg.norm(foot_grf[2] - target_grf) / grf_tolerance) ** 2)
        right_grf = np.exp(-(np.linalg.norm(foot_grf[5] - target_grf) / grf_tolerance) ** 2)

        # reward is only activated when both feet are down
        r_grf = 0.5 * left_grf + 0.5 * right_grf

        # B. Target/Imitation Reward
        target_pos = np.array([0.0, 0.0, 1.01,
                               1.0, 0.0, 0.0, 0.0,
                               0.0045, 0.0, 0.4973,
                               0.9784830934748516, -0.016399716640763992, 0.017869691242100763, -0.2048964597373501,
                               -1.1997, 0.0, 1.4267, 0.0, -1.5244, 1.5244, -1.5968,
                               -0.0045, 0.0, 0.4973,
                               0.978614127766972, 0.0038600557257107214, -0.01524022001550036, -0.20510296096975877,
                               -1.1997, 0.0, 1.4267, 0.0, -1.5244, 1.5244, -1.5968])
        r_target_joint_pos = np.exp(-np.linalg.norm(qpos - target_pos) ** 2)

        # Total Reward
        reward = (rw[0] * r_pose
                  + rw[1] * r_com_pos
                  + rw[2] * r_com_vel
                  + rw[3] * r_foot_placement
                  + rw[4] * r_fp_orient
                  + rw[5] * r_grf
                  + rw[6] * r_target_joint_pos)

        # ./train.py -r 0.3 --eps_steps 30 --training_steps 1e6 --save --tensorboard --tag "TargetPosition[]"

        print('Pose [{:.3f}], CoM [{:.3f}, {:.3f}], Foot [{:.3f}, {:.3f}], GRF[{:.3f}] Target [{:.3f}]'.format(r_pose,
                                                                                                               r_com_pos,
                                                                                                               r_com_vel,
                                                                                                               r_foot_placement,
                                                                                                               r_fp_orient,
                                                                                                               r_grf,
                                                                                                               r_target_joint_pos))

        return reward

    def compute_cost(self, qpos, foot_pos, foot_grf, cw=(0.3, 0.1, 0., 0.5)):
        # 1. Ground Contact (At least 1 foot must be on the ground)
        c_contact = 1. if (foot_grf[2] + foot_grf[5]) == 0. else 0.

        # 2. Power Consumption
        # Specs taken from RoboDrive datasheet for ILM 115x50

        # in Newton-meters
        max_motor_torques = np.array([4.66, 4.66, 12.7, 12.7, 0.99,
                                      4.66, 4.66, 12.7, 12.7, 0.99])

        # in Watts
        power_loss_at_max_torque = np.array([19.3, 19.3, 20.9, 20.9, 5.4,
                                             19.3, 19.3, 20.9, 20.9, 5.4])

        gear_ratios = np.array([25, 25, 16, 16, 50,
                                25, 25, 16, 16, 50])

        # calculate power loss constants
        power_loss_constants = power_loss_at_max_torque / np.square(max_motor_torques)

        # get output torques and velocities
        output_torques = np.array(self.cassie_state.motor.torque[:10])
        output_velocity = np.array(self.cassie_state.motor.velocity[:10])

        # calculate input torques
        input_torques = output_torques / gear_ratios

        # get power loss of each motor
        power_losses = power_loss_constants * np.square(input_torques)

        # calculate motor power for each motor
        motor_powers = np.amax(np.diag(output_torques).dot(output_velocity.reshape(10, 1)), initial=0, axis=1)

        # estimate power
        power_estimate = np.sum(motor_powers) + np.sum(power_losses)

        power_threshold = 150  # Watts (Positive Work only)
        c_power = 1. / (1. + np.exp(-(power_estimate - power_threshold)))

        # 3. Smooth Torques Cost
        c_smooth_actions = 1 - np.exp(-np.sum(input_torques) ** 2)

        # 4. Falling
        c_fall = 1 if qpos[2] < self.min_height else 0

        # Total Cost
        cost = cw[0] * c_contact + cw[1] * c_power + cw[2] * c_smooth_actions + cw[3] * c_fall

        return cost

    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        pos = np.copy(self.trajectory.qpos[phase * self.simrate])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter

        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ###### Setting variable speed  #########
        pos[0] *= self.speed
        pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.speed
        ######                          ########

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        vel = np.copy(self.trajectory.qvel[phase * self.simrate])
        vel[0] *= self.speed

        return pos, vel

    def get_full_state(self):

        ext_state = [self.speed]

        if self.clock_based:
            # Clock is muted for standing
            clock = [0., 0.]

            # Concatenate clock with ext_state
            ext_state = np.concatenate((clock, ext_state))

        if self.state_est:
            # Use state estimator
            robot_state = np.concatenate([
                [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height],  # pelvis height
                self.cassie_state.pelvis.orientation[:],  # pelvis orientation
                self.cassie_state.motor.position[:],  # actuated joint positions

                self.cassie_state.pelvis.translationalVelocity[:],  # pelvis translational velocity
                self.cassie_state.pelvis.rotationalVelocity[:],  # pelvis rotational velocity
                self.cassie_state.motor.velocity[:],  # actuated joint velocities

                self.cassie_state.pelvis.translationalAcceleration[:],  # pelvis translational acceleration

                self.cassie_state.joint.position[:],  # unactuated joint positions
                self.cassie_state.joint.velocity[:]  # unactuated joint velocities
            ])

            # robot_state = np.concatenate([
            #     self.cassie_state.pelvis.orientation[:],  # pelvis orientation
            #     self.cassie_state.pelvis.rotationalVelocity[:],  # pelvis rotational velocity
            #
            #     self.cassie_state.motor.position[:],  # actuated joint positions
            #     self.cassie_state.motor.velocity[:],  # actuated joint velocities
            #
            #     self.cassie_state.joint.position[:],  # unactuated joint positions
            #     self.cassie_state.joint.velocity[:]  # unactuated joint velocities
            # ])

            # Concatenate robot_state to ext_state
            ext_state = np.concatenate((robot_state, ext_state))

        return ext_state

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, self.config)

        self.vis.draw(self.sim)
