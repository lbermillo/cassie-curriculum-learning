import os
import gym
import random
import numpy as np

from math import floor
from copy import deepcopy
from cassie.utils.quaternion_function import *
from cassie.trajectory import CassieTrajectory
from cassie.utils.power_estimation import estimate_power
from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis


# Creating the Standing Environment
class CassieEnv:

    def __init__(self, simrate=50, clock_based=True, state_est=True,
                 reward_cutoff=0.3, target_action_weight=1.0, target_height=0.9, target_speed=(0, 0, 0),
                 forces=(0, 0, 0), force_fq=100, min_height=0.4, max_height=3.0, fall_threshold=0.3,
                 min_speed=(0, 0, 0), max_speed=(1, 1, 1), power_threshold=150, reduced_input=False, debug=False,
                 config="cassie/cassiemujoco/cassie.xml", traj='walking', writer=None):

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
        self.fall_threshold = fall_threshold
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.target_height = target_height
        self.target_speed = target_speed
        self.power_threshold = power_threshold
        self.reduced_input = reduced_input
        self.debug = debug
        self.writer = writer

        # Cassie properties
        self.mass   = self.sim.get_body_mass()
        self.weight = np.sum(self.mass) * 9.81

        # L/R midfoot offset (https://github.com/agilityrobotics/agility-cassie-doc/wiki/Toe-Model)
        # Already included in cassie_mujoco
        # self.midfoot_offset = np.array([0.1762, 0.05219, 0., 0.1762, -0.05219, 0.])
        self.midfoot_offset = np.array([0., 0., 0., 0., 0., 0.])

        # action offset so that the policy can learn faster and prevent standing on heels
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968,
                                0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.offset_weight = target_action_weight

        # tracking various variables for reward funcs
        self.l_foot_frc = np.zeros(3)
        self.r_foot_frc = np.zeros(3)
        self.l_foot_vel = np.zeros(3)
        self.r_foot_vel = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)

        self.timestep = 0
        self.total_steps = 0

        # Initial Actions
        self.P = np.array([100, 100, 88, 96, 50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.cassie_state = state_out_t()
        self.simrate = simrate

        dirname = os.path.dirname(__file__)
        if traj == "walking":
            traj_path = os.path.join(dirname, "..", "trajectory", "stepdata.bin")

        elif traj == "stepping":
            # traj_path = os.path.join(dirname, "trajectory", "spline_stepping_traj.pkl")
            traj_path = os.path.join(dirname, "..", "trajectory", "more-poses-trial.bin")

        self.trajectory = CassieTrajectory(traj_path)

        # total number of phases
        self.phaselen = floor(len(self.trajectory) / self.simrate) - 1

        # See include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        # Initialize Observation and Action Spaces
        self.observation_space = np.zeros(len(self.get_full_state()))
        self.action_space = gym.spaces.Box(-1. * np.ones(10), 1. * np.ones(10), dtype=np.float32)

        # Action tracking to penalize large action changes
        self.previous_action = np.zeros(10)

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

        # random changes to orientation
        # if np.random.randint(300) == 0:
        #     self.orient_add += np.random.uniform(-self.max_orient_change, self.max_orient_change)

        # simulating delays
        # simrate = self.simrate + np.random.randint(-10, 0)
        simrate = self.simrate

        # reset mujoco tracking variables
        foot_pos = np.zeros(6)
        self.l_foot_frc = np.zeros(3)
        self.r_foot_frc = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)

        for _ in range(simrate):
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
        reward = self.compute_reward(qpos, qvel, foot_pos, foot_grf) \
                 - self.compute_cost(qpos, action, foot_vel, foot_grf) if height_in_bounds else 0.0

        # Done Condition
        done = True if not height_in_bounds or reward < self.reward_cutoff else False

        # Update timestep counter
        self.timestep += 1
        self.total_steps += 1

        return state, reward, done, {}

    def reset(self, reset_ratio=0, use_phase=False):
        # reset variables
        self.timestep = 0
        self.sim.full_reset()
        self.reset_cassie_state()
        self.previous_action = np.zeros(10)

        # reset with perturbations and/or from a different stance (use_phase=True)
        if np.random.rand() < reset_ratio:

            # initialize qvel and get desired xyz velocities
            qvel = np.copy(self.sim.qvel())
            x_speed = random.randint(int(self.min_speed[0] * 10), int(self.max_speed[0] * 10)) / 10.
            y_speed = random.randint(int(self.min_speed[1] * 10), int(self.max_speed[1] * 10)) / 10.
            z_speed = random.randint(int(self.min_speed[2] * 10), int(self.max_speed[2] * 10)) / 10.

            if use_phase and np.random.rand() < reset_ratio:
                # get the corresponding state from the reference trajectory for the current phase
                qpos, qvel = self.get_ref_state(random.randint(0, self.phaselen), x_speed)

                # set joint positions
                self.sim.set_qpos(qpos)

            # modify qvel to have desired xyz joint velocities
            qvel[:3] = [x_speed, y_speed, z_speed]

            # set joint velocities
            self.sim.set_qvel(qvel)

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

    def compute_reward(self, qpos, qvel, foot_pos, foot_grf, rw=(0.2, 0.2, 0.1, 0.1, 0.2, 0.2)):

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
        com_position_coeff = 25

        # 2a. Horizontal Position Component (target position is the center of the support polygon)
        xy_target_pos = np.array([0.5 * (foot_pos[0] + foot_pos[3]),
                                  0.5 * (foot_pos[1] + foot_pos[4])])

        xy_com_pos = np.exp(-com_position_coeff * np.linalg.norm(qpos[:2] - xy_target_pos) ** 2)

        # 2b. Vertical Position Component (robot should stand upright and maintain a certain height)
        z_com_pos = np.exp(-com_position_coeff * (qpos[2] - self.target_height) ** 2)

        r_com_pos = 0.5 * xy_com_pos + 0.5 * z_com_pos

        # 3. CoM Velocity Modulation
        # Derive horizontal target speed from CP formula and vertical target speed should be 0
        # where x_cp = x_com + x_vel * sqrt(z / 9.81) and y_cp can be captured using the same formula

        # xy_target_speed  = (xy_target_pos - [qpos[0], qpos[1]]) / np.sqrt(qpos[2] / 9.81)
        # com_target_speed = np.concatenate((xy_target_speed, [0]))
        #
        # r_com_vel = np.exp(-np.linalg.norm(qvel[:3] - com_target_speed) ** 2)

        r_com_vel = np.exp(-np.linalg.norm(qvel[:3] - np.zeros(3)) ** 2)

        # 4. Foot Placement
        foot_placement_coeff = 100

        # 4a. Foot Alignment
        r_feet_align = np.exp(-foot_placement_coeff * (foot_pos[0] - foot_pos[3]) ** 2)

        # 4b. Feet Width
        target_width = 0.18  # 18 cm seems to be optimal
        feet_width = np.linalg.norm([foot_pos[1], foot_pos[4]])
        r_foot_width = np.exp(-foot_placement_coeff * (feet_width - target_width) ** 2)

        r_foot_placement = 0.5 * r_feet_align + 0.5 * r_foot_width

        # 5. Foot/Pelvis Orientation
        fp_orientation_coeff = 5e2
        _, _, pelvis_yaw = quaternion2euler(qpos[3:7])
        foot_yaw = np.array([qpos[8], qpos[22]])
        left_foot_orient  = np.exp(-fp_orientation_coeff * (foot_yaw[0] - pelvis_yaw) ** 2)
        right_foot_orient = np.exp(-fp_orientation_coeff * (foot_yaw[1] - pelvis_yaw) ** 2)

        r_fp_orient = 0.5 * left_foot_orient + 0.5 * right_foot_orient

        # 6. Ground Force Modulation
        target_grf = self.weight / 2.

        left_grf  = np.exp(-2e-3 * (np.linalg.norm(foot_grf[2] - target_grf)) ** 2)
        right_grf = np.exp(-2e-3 * (np.linalg.norm(foot_grf[5] - target_grf)) ** 2)

        r_grf = 0.5 * left_grf + 0.5 * right_grf if np.sum([qvel[0], qvel[1]]) < 0.1 else 0

        # Total Reward
        reward = (rw[0] * r_pose
                  + rw[1] * r_com_pos
                  + rw[2] * r_com_vel
                  + rw[3] * r_foot_placement
                  + rw[4] * r_fp_orient
                  + rw[5] * r_grf)

        if self.writer is not None and self.debug:
            # log episode reward to tensorboard
            self.writer.add_scalar('env_reward/pose', r_pose, self.total_steps)
            self.writer.add_scalar('env_reward/com_pos', r_com_pos, self.total_steps)
            self.writer.add_scalar('env_reward/com_vel', r_com_vel, self.total_steps)
            self.writer.add_scalar('env_reward/foot_placement', r_foot_placement, self.total_steps)
            self.writer.add_scalar('env_reward/foot_orientation', r_fp_orient, self.total_steps)
            self.writer.add_scalar('env_reward/grf', r_grf, self.total_steps)
        elif self.debug:
            print('[{}] Rewards: Pose [{:.3f}], CoM [{:.3f}, {:.3f}], Foot [{:.3f}, {:.3f}], GRF[{:.3f}]]'.format(self.timestep,
                                                                                                                  r_pose,
                                                                                                                  r_com_pos,
                                                                                                                  r_com_vel,
                                                                                                                  r_foot_placement,
                                                                                                                  r_fp_orient,
                                                                                                                  r_grf, ))

        return reward

    def compute_cost(self, qpos, action, foot_vel, foot_grf, cw=(0.3, 0.2, 0.1, 0.1, 0.1, 0.)):
        # 1. Falling
        c_fall = 1 if qpos[2] < self.target_height - self.fall_threshold else 0

        # 2. Ground Contact (At least 1 foot must be on the ground)
        c_contact = 1 if (foot_grf[2] + foot_grf[5]) == 0 else 0

        # 3. Power Consumption
        power_estimate, power_info = estimate_power(self.cassie_state.motor.torque[:10], self.cassie_state.motor.velocity[:10])

        # power_coeff = 1e-5
        power_coeff = np.min([self.total_steps * (1e-5 / 1e6), 1e-5])
        c_power = 1. - np.exp(-power_coeff * power_estimate ** 2)

        # 4. Action Change Cost
        # actions_coeff = 1e3
        actions_coeff = np.min([self.total_steps * (1e3 / 7e6), 1e3])
        c_actions = 1 - np.exp(-actions_coeff * np.linalg.norm(action - self.previous_action) ** 2)

        # 5. Toe Cost
        # toe_coeff = 3e-6
        toe_coeff = np.min([self.total_steps * (3e-6 / 5e6), 3e-6])
        c_toe = 1 - np.exp(-toe_coeff * np.linalg.norm([self.cassie_state.motor.torque[4],
                                                        self.cassie_state.motor.torque[9]]) ** 4)

        # 6. Foot Drag (X-Y GRFs)
        foot_x_drag = (1 - np.exp(-np.linalg.norm([foot_grf[0], foot_grf[3]]) ** 2)) \
            * (1 - np.exp(-100 * np.linalg.norm([foot_vel[0], foot_vel[3]]) ** 2))

        foot_y_drag = (1 - np.exp(-np.linalg.norm([foot_grf[1], foot_grf[4]]) ** 2)) \
            * (1 - np.exp(-500 * np.linalg.norm([foot_vel[1], foot_vel[4]]) ** 2))

        c_drag = 0.75 * foot_y_drag + 0.25 * foot_x_drag

        # Update previous torque with current one
        self.previous_action = action

        # Total Cost
        cost = cw[0] * c_fall + cw[1] * c_contact + cw[2] * c_power + cw[3] * c_actions \
            + cw[4] * c_toe + cw[5] * c_drag

        if self.writer is not None and self.debug:
            # log episode reward to tensorboard
            self.writer.add_scalar('env_cost/fall', c_fall, self.total_steps)
            self.writer.add_scalar('env_cost/foot_contact', c_contact, self.total_steps)
            self.writer.add_scalar('env_cost/power_consumption', c_power, self.total_steps)
            self.writer.add_scalar('env_cost/action_change', c_actions, self.total_steps)
            self.writer.add_scalar('env_cost/foot_drag', c_drag, self.total_steps)
            self.writer.add_scalar('env_cost/toe_usage', c_toe, self.total_steps)
        elif self.debug:
            print('Costs:\t Fall [{:.3f}], Contact [{:.3f}], Power [{:.3f}], Action Change [{:.3f}], '
                  'Toe [{:.3f}], Drag [{:.3f}]\n'.format(c_fall, c_contact, c_power, c_actions, c_toe, c_drag))

        return cost

    def get_ref_state(self, phase, speed):
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

    def rotate_to_orient(self, vec):
        quaternion = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if len(vec) == 3:
            return rotate_by_quaternion(vec, iquaternion)

        elif len(vec) == 4:
            new_orient = quaternion_product(iquaternion, vec)
            if new_orient[0] < 0:
                new_orient = -new_orient
            return new_orient

    def get_full_state(self):

        ext_state = [self.target_speed[0]]

        if self.clock_based:
            # Clock is muted for standing
            clock = [0., 0.]

            # Concatenate clock with ext_state
            ext_state = np.concatenate((clock, ext_state))

        if self.state_est:
            # pelvis_orientation = self.rotate_to_orient(self.cassie_state.pelvis.orientation[:])

            # Use state estimator
            robot_state = np.concatenate([

                # Pelvis States
                self.cassie_state.pelvis.orientation[:],
                self.cassie_state.pelvis.rotationalVelocity[:],

                # Motor States
                self.cassie_state.motor.position[:],
                self.cassie_state.motor.velocity[:],

                # Foot States
                # self.cassie_state.leftFoot.position[:],
                # self.cassie_state.rightFoot.position[:],

            ])

            if not self.reduced_input:
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

            # Concatenate robot_state to ext_state
            ext_state = np.concatenate((robot_state, ext_state))

        return ext_state

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, self.config)

        self.vis.draw(self.sim)
