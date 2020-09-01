import os
import gym
import random
import numpy as np

from math import floor
from copy import deepcopy
from cassie.trajectory import CassieTrajectory
from cassie.quaternion_function import quaternion2euler
from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis


# Creating the Walking Environment
class CassieEnv:

    def __init__(self, simrate=60, clock_based=True, state_est=True,
                 reward_cutoff=0.3, target_action_weight=1.0, target_height=0.9, forces=(0, 0, 0), force_fq=100,
                 min_height=0.6, max_height=3.0, fall_height=0.4, min_speed=0.5, max_speed=2.0, power_threshold=150, debug=False,
                 config="cassie/cassiemujoco/cassie.xml", traj='walking'):

        # Using CassieSim
        self.config = config
        self.sim = CassieSim(self.config)
        self.vis = None

        # Initialize parameters
        self.clock_based = clock_based
        self.state_est = state_est
        self.reward_cutoff = reward_cutoff
        self.forces = forces
        self.force_fq = force_fq
        self.min_height = min_height
        self.max_height = max_height
        self.fall_height = fall_height
        # TODO: in future development, format speeds to [x, y, z]
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.target_height = target_height
        self.power_threshold = power_threshold
        self.debug = debug

        # Cassie properties
        self.mass = np.sum(self.sim.get_body_mass())
        self.weight = self.mass * 9.81

        # L/R midfoot offset (https://github.com/agilityrobotics/agility-cassie-doc/wiki/Toe-Model)
        # Already included in cassie_mujoco
        # self.midfoot_offset = np.array([0.1762, 0.05219, 0., 0.1762, -0.05219, 0.])
        self.midfoot_offset = np.array([0., 0., 0., 0., 0., 0.])

        # action offset so that the policy can learn faster and prevent standing on heels
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968,
                                0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.offset_weight = target_action_weight

        # Initialize Observation and Action Spaces (+1 is for speed input)
        # TODO: Future development +3 for speeds [x, y, z], +1 for orientation,
        #       and input reduction to only directly measured sensors
        self.observation_size = 40 + 1

        if self.clock_based:
            # Add two more inputs for right and left clocks
            self.observation_size += 2

        if self.state_est:
            self.observation_size += 6

        self.observation_space = np.zeros(self.observation_size)
        self.action_space = gym.spaces.Box(-1. * np.ones(10), 1. * np.ones(10))

        # tracking various variables for reward funcs
        self.l_foot_frc = np.zeros(3)
        self.r_foot_frc = np.zeros(3)
        self.l_foot_vel = np.zeros(3)
        self.r_foot_vel = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)

        self.timestep = 0

        # Initial Actions
        self.P = np.array([100, 100, 88, 96, 50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.cassie_state = state_out_t()
        self.simrate = simrate
        self.speed = 0

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

        foot_pos = np.zeros(6)
        self.sim.foot_pos(foot_pos)
        prev_foot = deepcopy(foot_pos)
        self.u = pd_in_t()

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

        # Update tracking variables
        self.sim.foot_pos(foot_pos)
        self.l_foot_vel = (foot_pos[0:3] - prev_foot[0:3]) / 0.0005
        self.r_foot_vel = (foot_pos[3:6] - prev_foot[3:6]) / 0.0005

    def step(self, action):

        if self.timestep % self.force_fq == 0:
            # Apply perturbations to the pelvis
            self.sim.apply_force([random.choice([-self.forces[0], self.forces[0]]),
                                  random.choice([-self.forces[1], self.forces[1]]),
                                  random.choice([-self.forces[2], self.forces[2]]),
                                  0,
                                  0])

        # reset mujoco tracking variables
        foot_pos = np.zeros(6)
        self.l_foot_frc = np.zeros(3)
        self.r_foot_frc = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)

        for _ in range(self.simrate):
            self.step_simulation(action)

            # Foot Force Tracking
            foot_forces = self.sim.get_foot_forces()
            self.l_foot_frc += foot_forces[0:3]
            self.r_foot_frc += foot_forces[3:6]

            # Relative Foot Position tracking
            self.sim.foot_pos(foot_pos)
            self.l_foot_pos += foot_pos[0:3]
            self.r_foot_pos += foot_pos[3:6]

        # CoM Position and Velocity
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        # Get the average foot positions and forces
        self.l_foot_frc /= self.simrate
        self.r_foot_frc /= self.simrate
        self.l_foot_pos /= self.simrate
        self.r_foot_pos /= self.simrate

        # Create lists for foot positions, velocities, and forces
        foot_pos = np.append(self.l_foot_pos, self.r_foot_pos)
        foot_vel = np.append(self.l_foot_vel, self.r_foot_vel)
        foot_grf = np.append(self.l_foot_frc, self.r_foot_frc)

        # Current State
        state = self.get_full_state()

        # Early termination condition
        height_in_bounds = self.min_height < self.sim.qpos()[2] < self.max_height

        # Current Reward
        reward = self.compute_reward(qpos, qvel, foot_pos) \
                 - self.compute_cost(qpos, foot_grf) if height_in_bounds else 0.0

        # Done Condition
        done = True if not height_in_bounds or reward < self.reward_cutoff else False

        # Update timestep counter
        self.timestep += 1

        return state, reward, done, {}

    def reset(self, phase=None, phase_reset_ratio=0.7):
        # reset variables
        self.timestep = 0
        self.speed = random.randint(int(self.min_speed * 10), int(self.max_speed * 10)) / 10.

        # phase reset ratio = 1 means use phase reset of the time while 0 means don't use phase reset
        if np.random.rand() < phase_reset_ratio:
            self.phase = int(phase) if phase is not None else random.randint(0, self.phaselen)

            # get the corresponding state from the reference trajectory for the current phase
            qpos, qvel = self.get_ref_state(self.phase)

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(qvel)

        else:
            self.sim.full_reset()
            self.reset_cassie_state()

        return self.get_full_state()

    def compute_reward(self, qpos, qvel, foot_pos, rw=(0.15, 0.15, 0.3, 0.2, 0.2, 0, 0), multiplier=500):

        left_foot_pos = foot_pos[:3]
        right_foot_pos = foot_pos[3:]

        # midfoot position
        foot_pos = np.concatenate([left_foot_pos - self.midfoot_offset[:3], right_foot_pos - self.midfoot_offset[3:]])

        # A. Task Rewards

        # 1. Pelvis Orientation [https://math.stackexchange.com/questions/90081/quaternion-distance]
        target_pose = np.array([1, 0, 0, 0])
        pose_error = 1 - np.inner(qpos[3:7], target_pose) ** 2

        r_pose = np.exp(-1e5 * pose_error ** 2)

        # 2. CoM Position Modulation
        # 2a. Horizontal Position Component (target position is the center of the support polygon)
        xy_target_pos = np.array([0.5 * (foot_pos[0] + foot_pos[3]),
                                  0.5 * (foot_pos[1] + foot_pos[4])])

        xy_com_pos = np.exp(-np.sum(qpos[:2] - xy_target_pos) ** 2)

        # 2b. Vertical Position Component (robot should stand upright and maintain a certain height)
        height_thresh = 0.1  # m = 10 cm
        z_target_pos = self.target_height

        if qpos[2] < z_target_pos - height_thresh:
            z_com_pos = np.exp(-100 * (qpos[2] - (z_target_pos - height_thresh)) ** 2)
        elif qpos[2] > z_target_pos + 0.1:
            z_com_pos = np.exp(-100 * (qpos[2] - (z_target_pos + height_thresh)) ** 2)
        else:
            z_com_pos = 1.

        r_com_pos = 0.5 * xy_com_pos + 0.5 * z_com_pos

        # 3. CoM Velocity Modulation
        target_speed = np.array([self.speed, 0, 0])
        r_com_vel = np.exp(-np.linalg.norm(target_speed - qvel[:3]) ** 2)

        # 4. Feet Width
        width_thresh = 0.02     # m = 2 cm
        target_width = 0.18     # m = 18 cm seems to be optimal
        feet_width = np.linalg.norm([foot_pos[1], foot_pos[4]])

        if feet_width < target_width - width_thresh:
            r_foot_width = np.exp(-multiplier * (feet_width - (target_width - width_thresh)) ** 2)
        elif feet_width > target_width + width_thresh:
            r_foot_width = np.exp(-multiplier * (feet_width - (target_width + width_thresh)) ** 2)
        else:
            r_foot_width = 1.

        # 5. Foot/Pelvis Orientation (may need to be revisited when doing turning)
        _, _, pelvis_yaw = quaternion2euler(qpos[3:7])
        foot_yaw = np.array([qpos[8], qpos[22]])
        left_foot_orient = np.exp(-multiplier * (foot_yaw[0] - pelvis_yaw) ** 2)
        right_foot_orient = np.exp(-multiplier * (foot_yaw[1] - pelvis_yaw) ** 2)

        r_fp_orient = 0.5 * left_foot_orient + 0.5 * right_foot_orient

        # 6. TODO: Symmetric Actions (only valid for non-turning actions)

        # Total Reward
        reward = (rw[0] * r_pose
                  + rw[1] * r_com_pos
                  + rw[2] * r_com_vel
                  + rw[3] * r_foot_width
                  + rw[4] * r_fp_orient)

        if self.debug:
            print('Pose [{:.3f}], CoM [{:.3f}, {:.3f}], Foot [{:.3f}, {:.3f}]]'.format(r_pose,
                                                                                       r_com_pos,
                                                                                       r_com_vel,
                                                                                       r_foot_width,
                                                                                       r_fp_orient))

        return reward

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

    def compute_cost(self, qpos, foot_grf, cw=(0, 0.1, 0.5)):
        # 1. Ground Contact (At least 1 foot must be on the ground)
        # TODO: Only valid for walking gaits
        c_contact = 1 if (foot_grf[2] + foot_grf[5]) == 0 else 0

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

        c_power = 1. / (1. + np.exp(-(power_estimate - self.power_threshold)))

        # 3. Falling
        c_fall = 1 if qpos[2] < self.fall_height else 0

        # Total Cost
        cost = cw[0] * c_contact + cw[1] * c_power + cw[2] * c_fall

        return cost

    def get_ref_state(self, phase=None, speed=None):
        phase = self.phase if phase is None else phase
        speed = self.speed if speed is None else speed

        if phase > self.phaselen:
            phase = 0

        pos = np.copy(self.trajectory.qpos[phase * self.simrate])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter

        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ###### Setting variable speed  #########
        pos[0] *= speed
        pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * speed
        ######                          ########

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        vel = np.copy(self.trajectory.qvel[phase * self.simrate])
        vel[0] *= speed

        return pos, vel

    def get_full_state(self):

        ext_state = [self.speed]

        if self.clock_based:
            # TODO: Change this to the proper clock cycle
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
