import os
import gym
import random
import numpy as np

from math import floor
from copy import deepcopy
from cassie.utils.mirror import mirror
from cassie.utils.quaternion_function import *
from cassie.trajectory import CassieTrajectory
from cassie.utils.power_estimation import estimate_power
from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from cassie.utils.dynamics_randomization import randomize_mass, randomize_friction


class CassieEnv:

    def __init__(self, simrate=50, clock_based=True,
                 reward_cutoff=0.3, target_action_weight=1.0, target_height=0.8, training_steps=1e6,
                 forces=(0, 0, 0), force_fq=100, min_height=0.4, max_height=3.0, max_orient=0, fall_threshold=0.3,
                 min_speed=(0, 0, 0), max_speed=(1, 1, 1), power_threshold=150, reduced_input=False, learn_PD=False,
                 debug=False, config="cassie/cassiemujoco/cassie.xml", traj='walking', writer=None):

        # Using CassieSim
        self.config = config
        self.sim = CassieSim(self.config)
        self.vis = None

        # Initialize parameters
        self.clock_based = clock_based
        self.reward_cutoff = reward_cutoff
        self.training_steps = training_steps
        self.forces = forces
        self.force_fq = force_fq
        self.min_height = min_height
        self.max_height = max_height
        self.fall_threshold = fall_threshold
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_orient = max_orient
        self.target_height = target_height
        self.target_speed = np.zeros(3)
        self.target_orientation = np.zeros(3)
        self.power_threshold = power_threshold
        self.reduced_input = reduced_input
        self.learn_PD = learn_PD
        self.debug = debug
        self.writer = writer

        # Encoder noise
        self.encoder_noise = 0.01
        self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
        self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)

        # Get simulation properties for dynamics randomization and rewards
        self.mass   = self.sim.get_body_mass()
        self.weight = np.sum(self.mass) * 9.81
        self.friction = self.sim.get_geom_friction()

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

        # TODO: learn_PD
        # Initial Actions
        self.P = np.array([100., 100., 88., 96., 50.]) if not self.learn_PD else np.random.randint(low=0, high=100, size=10)
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0]) if not self.learn_PD else np.random.randint(low=0, high=100, size=10)

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

        # TODO: learn_PD
        # Set number of actions based on PD gains are learned or not
        # (30 if learning PD gains = 10 jont positions + 2 gains * 10 joints)
        num_actions = 30 if self.learn_PD else 10

        # num_actions = 6 # TODO: Removing policy control of hip yaw and hip roll

        # Action and velocity tracking for large changes
        self.previous_action = np.zeros(num_actions)
        self.previous_velocity = self.cassie_state.motor.velocity[:]

        # Initialize Observation and Action Spaces
        self.observation_space = np.zeros(len(self.get_full_state()))
        self.action_space = gym.spaces.Box(-1. * np.ones(num_actions), 1. * np.ones(num_actions), dtype=np.float32)

        # Used for eval
        self.test = False

    def eval(self):
        self.test = True

    def train(self):
        self.test = False

    def close(self):
        if self.vis is not None:
            del self.vis
            self.vis = None

    def step_simulation(self, action):

        # TODO: learn_PD
        if self.learn_PD:
            action, self.P, self.D = action[:10], np.abs(action[10:20] * 100), np.abs(action[20:] * 10)

        # TODO: Create Target Action
        # action = np.array([0., 0., action[0], action[1], action[2],
        #                    0., 0., action[3], action[4], action[5]])
        # Uncomment to debug hip roll and yaw motors
        # action[0] = 0.
        # action[1] = 0.
        # action[5] = 0.
        # action[6] = 0.
        target = action + (self.offset_weight * self.offset) - self.motor_encoder_noise

        foot_pos = np.zeros(6)
        self.sim.foot_pos(foot_pos)
        prev_foot = deepcopy(foot_pos)
        self.u = pd_in_t()

        # Apply Action
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i] if not self.learn_PD else self.P[i + 5]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i] if not self.learn_PD else self.D[i + 5]

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

        # random changes to target yaw of the pelvis
        if np.random.rand() < 0.1:
            self.target_orientation[2] += np.random.uniform(-self.max_orient, self.max_orient)

        # simulating delays
        simrate = self.simrate + np.random.randint(-10, 10)
        # simrate = self.simrate

        # reset mujoco tracking variables
        foot_pos = np.zeros(6)
        self.l_foot_frc = np.zeros(3)
        self.r_foot_frc = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)

        for rate in range(simrate):
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

        # TODO: print qpos
        # print(qpos)

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
        reward = self.compute_reward(qpos, qvel, foot_pos, foot_grf) - self.compute_cost(action) if height_in_bounds \
            else self.compute_cost(action)

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
        self.previous_action = np.zeros(self.action_space.shape[0])
        self.previous_velocity = self.cassie_state.motor.velocity[:]

        # reset target variables
        self.target_orientation = np.zeros(3)
        self.target_speed[0] = random.randint(int(self.min_speed[0] * 10), int(self.max_speed[0] * 10)) / 10.
        self.target_speed[1] = random.randint(int(self.min_speed[1] * 10), int(self.max_speed[1] * 10)) / 10.
        self.target_speed[2] = random.randint(int(self.min_speed[2] * 10), int(self.max_speed[2] * 10)) / 10.

        # TODO: learn_PD
        if self.learn_PD:
            self.P = np.random.randint(low=0, high=100, size=10)
            self.D = np.random.randint(low=0, high=10,  size=10)

        if not self.test:
            # TODO: randomize mass and recalculate weight
            self.sim.set_body_mass(randomize_mass(self.mass, low=0.9, high=1.1))
            self.weight = np.sum(self.sim.get_body_mass()) * 9.81

            # TODO: randomize ground friction
            self.sim.set_geom_friction(randomize_friction(self.friction, low=0.9, high=1.1))

        # TODO: randomize initial height
        self.randomize_init_height()

        # TODO: joint noise error
        self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
        self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)

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

        # Take a step to set cassie_state
        self.cassie_state = self.sim.step_pd(self.u)

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

    def randomize_init_height(self):
        low_start = [0.02477726, -0.01568609, 0.61515595, 0.99910102, -0.03834618, -0.01492016,
                     -0.01020324, 0.10320576, -0.01822352, 0.78221161, 0.77875311, -0.00615072,
                     0.07485685, -0.62281796, -2.10276909, -0.06401924, 2.5159155, -0.06195463,
                     -1.93171504, 1.91307376, -2.03487245, 0.1189546, 0.03197041, 0.75972677,
                     0.79644527, -0.002866, -0.06312636, -0.60139985, -2.03505813, -0.07590401,
                     2.47309159, -0.06861883, -1.92336564, 1.90473564, -2.02667971]

        mid_start = [-2.83004740e-03, -2.29114732e-02, 7.07065845e-01, 9.99894779e-01,
                     1.10562740e-02, -6.12652366e-03, 7.11723211e-03, 1.50099449e-02,
                     1.64266778e-02, 8.24133043e-01, 8.37481179e-01, -9.69445402e-03,
                     6.13690062e-02, -5.42922773e-01, -1.90945471e+00, -4.85589866e-02,
                     2.26580429e+00, -4.62971329e-02, -1.91282419e+00, 1.89426871e+00,
                     -2.01747242e+00, 2.31781351e-03, 1.44250744e-02, 8.12155963e-01,
                     8.36154471e-01, -1.00090471e-03, -5.47184822e-02, -5.45756894e-01,
                     -1.90783323e+00, -5.71023096e-02, 2.28292790e+00, -5.06616206e-02,
                     -1.90963150e+00, 1.89109144e+00, -2.01460222e+00]

        high_start = [6.10130896e-04, -3.23304477e-04, 1.00659806e+00, 9.99977272e-01,
                      -1.24530409e-03, -6.60246460e-03, 5.58494911e-04, 1.06175814e-02,
                      6.16906927e-03, 4.46850777e-01, 9.81816364e-01, -1.65223535e-02,
                      1.59090533e-02, -1.88442409e-01, -1.13004694e+00, -4.11360961e-02,
                      1.44296047e+00, -2.39222365e-02, -1.46633365e+00, 1.44824750e+00,
                      -1.57126122e+00, -2.79549256e-03, 6.46317570e-03, 4.50763457e-01,
                      9.81461310e-01, 4.00951503e-03, -1.37635448e-02, -1.91123483e-01,
                      -1.15727612e+00, -1.67963434e-02, 1.43236886e+00, -2.01196652e-02,
                      -1.48618073e+00, 1.46805721e+00, -1.59446755e+00]

        init_height = random.choice([low_start, mid_start, high_start])

        self.sim.set_qpos(init_height)

    def compute_reward(self, qpos, qvel, foot_pos, foot_grf, rw=(0.25, 0.2, 0.2, 0.15, 0.2)):
        # rw=(pose, CoM pos, CoM vel, foot placement, foot/pelvis orientation)

        left_foot_pos  = foot_pos[:3]
        right_foot_pos = foot_pos[3:]

        # midfoot position
        foot_pos = np.concatenate([left_foot_pos - self.midfoot_offset[:3], right_foot_pos - self.midfoot_offset[3:]])

        # TODO: 1. Pelvis Orientation
        pelvis_orient_coeff = 10

        # convert quaternion values to euler [roll, pitch, yaw]
        pelvis_orient = quaternion2euler(qpos[3:7])

        r_pose = np.exp(-pelvis_orient_coeff * np.linalg.norm(pelvis_orient - self.target_orientation) ** 2)

        # 2. CoM Position Modulation
        xy_com_pos_coeff = 25
        z_com_pos_coeff = 10

        # 2a. Horizontal Position Component (target position is the center of the support polygon)
        xy_target_pos = np.array([0.5 * (foot_pos[0] + foot_pos[3]),
                                  0.5 * (foot_pos[1] + foot_pos[4])])

        xy_com_pos = np.exp(-xy_com_pos_coeff * np.linalg.norm(qpos[:2] - xy_target_pos) ** 2)

        # 2b. Vertical Position Component (robot should stand upright and maintain a certain height)
        height_thresh = 0.1  # 10 cm

        if qpos[2] < self.target_height - height_thresh:
            z_com_pos = np.exp(-z_com_pos_coeff * (qpos[2] - (self.target_height - height_thresh)) ** 2)
        elif qpos[2] > self.target_height + height_thresh:
            z_com_pos = np.exp(-z_com_pos_coeff * (qpos[2] - (self.target_height + height_thresh)) ** 2)
        else:
            z_com_pos = 1.

        r_com_pos = 0.8 * xy_com_pos + 0.2 * z_com_pos

        # 3. CoM Velocity Modulation
        com_vel_coeff = 50

        xy_target_vel = np.exp(-com_vel_coeff * np.linalg.norm(qvel[:2] - self.target_speed[:2]) ** 2)
        z_target_vel  = np.exp(-com_vel_coeff * np.linalg.norm(qvel[2]  - self.target_speed[2] ) ** 2)

        r_com_vel = 0.8 * xy_target_vel + 0.2 * z_target_vel

        # 4. Foot Placement
        foot_placement_coeff = 50

        # 4a. Foot Alignment
        r_feet_align = np.exp(-foot_placement_coeff * (foot_pos[0] - foot_pos[3]) ** 2)

        # 4b. Feet Width
        width_thresh = 0.02  # m = 2 cm
        target_width = 0.18  # m = 18 cm makes the support polygon a square
        feet_width = np.linalg.norm([foot_pos[1], foot_pos[4]])

        if feet_width < target_width - width_thresh:
            r_foot_width = np.exp(-foot_placement_coeff * (feet_width - (target_width - width_thresh)) ** 2)
        elif feet_width > target_width + width_thresh:
            r_foot_width = np.exp(-foot_placement_coeff * (feet_width - (target_width + width_thresh)) ** 2)
        else:
            r_foot_width = 1.

        r_foot_placement = 0. * r_feet_align + 1. * r_foot_width

        # 5. Foot/Pelvis Orientation
        fp_orientation_coeff = 15
        foot_yaw = np.array([qpos[8], qpos[22]])
        left_foot_orient  = np.exp(-fp_orientation_coeff * (foot_yaw[0] - pelvis_orient[2]) ** 2)
        right_foot_orient = np.exp(-fp_orientation_coeff * (foot_yaw[1] - pelvis_orient[2]) ** 2)

        r_fp_orient = 0.5 * left_foot_orient + 0.5 * right_foot_orient

        # 6. Ground Force Modulation (Not used, for reference only)
        target_grf = self.weight / 2.

        left_grf  = np.exp(-5e-4 * (np.linalg.norm(foot_grf[2] - target_grf)) ** 2)
        right_grf = np.exp(-5e-4 * (np.linalg.norm(foot_grf[5] - target_grf)) ** 2)

        r_grf = 0.5 * left_grf + 0.5 * right_grf

        # TODO: norm squared pose deviation from offset 10% - 20%
        # 7. Joint Position Reference (Offset is a standing position) use for hip roll and yaw
        # r_ref = np.exp(-np.linalg.norm(self.cassie_state.motor.position[:] - self.offset) ** 2)

        # Total Reward
        reward = (rw[0] * r_pose
                  + rw[1] * r_com_pos
                  + rw[2] * r_com_vel
                  + rw[3] * r_foot_placement
                  + rw[4] * r_fp_orient)

        if self.writer is not None and self.debug and not self.test:
            # log episode reward to tensorboard
            self.writer.add_scalar('env_reward/pose', r_pose, self.total_steps)
            self.writer.add_scalar('env_reward/com_pos', r_com_pos, self.total_steps)
            self.writer.add_scalar('env_reward/com_vel', r_com_vel, self.total_steps)
            self.writer.add_scalar('env_reward/foot_placement', r_foot_placement, self.total_steps)
            self.writer.add_scalar('env_reward/foot_orientation', r_fp_orient, self.total_steps)
        elif self.writer is None and self.debug:
            print('[{}] Rewards: Pose [{:.3f}], CoM [{:.3f}, {:.3f}], Foot [{:.3f}, {:.3f}], GRF[{:.3f}]]'.format(self.timestep,
                                                                                                                  r_pose,
                                                                                                                  r_com_pos,
                                                                                                                  r_com_vel,
                                                                                                                  r_foot_placement,
                                                                                                                  r_fp_orient,
                                                                                                                  r_grf,
                                                                                                                  ))

        return reward

    def compute_cost(self, action, cw=(0.1, 0.1, 0.1)):
        cost_coeff = 1 - np.exp(-(self.total_steps / self.training_steps) ** 2) if not self.test else 1.

        # 1. Power Consumption (Torque and Velocity)
        power_estimate, power_info = estimate_power(self.cassie_state.motor.torque[:10],
                                                    self.cassie_state.motor.velocity[:10])

        c_power = 1. - np.exp(-min(5e-5, cost_coeff) * power_estimate ** 2)

        # 2. Action Cost
        action_diff = np.subtract(self.previous_action, action)
        c_action = 1 - np.exp(-cost_coeff * np.linalg.norm(action_diff) ** 2)

        # 3. Motor Acceleration Cost
        motor_accel = np.subtract(self.previous_velocity, self.cassie_state.motor.velocity[:])
        c_maccel = 1 - np.exp(-min(5e-4, cost_coeff) * np.linalg.norm(motor_accel) ** 2)

        # Total Cost
        cost = cw[0] * c_power + cw[1] * c_action + cw[2] * c_maccel

        # Update previous variables
        self.previous_action = action
        self.previous_velocity = self.cassie_state.motor.velocity[:]

        if self.writer is not None and self.debug and not self.test:
            # log episode reward to tensorboard
            self.writer.add_scalar('env_cost/action_change', c_action, self.total_steps)
            self.writer.add_scalar('env_cost/motor_accel', c_maccel, self.total_steps)
            self.writer.add_scalar('env_cost/power', c_power, self.total_steps)
        elif self.writer is None and self.debug:
            print('Costs:\t Action Change [{:.3f}], Motor Acceleration [{:.3f}], Power [{:.3f}]\n'.format(
                c_action, c_maccel, c_power))

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
        quaternion = euler2quat(z=self.target_orientation[2], y=self.target_orientation[1], x=self.target_orientation[0])
        iquaternion = inverse_quaternion(quaternion)

        if len(vec) == 3:
            return rotate_by_quaternion(vec, iquaternion)

        elif len(vec) == 4:
            new_orient = quaternion_product(iquaternion, vec)
            if new_orient[0] < 0:
                new_orient = -new_orient
            return new_orient

    def get_full_state(self):

        # command consists of [forward velocity, lateral velocity, yaw rate]
        command = [self.target_speed[0], self.target_speed[1], self.target_orientation[2]]

        # create external state
        ext_state = np.concatenate((self.previous_action, command))

        if self.clock_based:
            # Clock is muted for standing
            clock = [0., 0.]

            # Concatenate clock with ext_state
            ext_state = np.concatenate((clock, ext_state))

        # Use state estimator
        robot_state = np.concatenate([

            # Pelvis States
            self.cassie_state.pelvis.orientation[:],
            self.cassie_state.pelvis.rotationalVelocity[:],

            # Motor States
            self.cassie_state.motor.position[:],
            self.cassie_state.motor.velocity[:],

        ])

        if not self.reduced_input:
            # Use state estimator
            robot_state = np.concatenate([
                # pelvis height
                [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height],

                # pelvis orientation
                self.rotate_to_orient(self.cassie_state.pelvis.orientation[:]),

                # pelvis linear/translational velocity
                self.rotate_to_orient(self.cassie_state.pelvis.translationalVelocity[:]),

                # pelvis rotational/angular velocity
                self.cassie_state.pelvis.rotationalVelocity[:],

                # joint positions w/ added noise
                self.cassie_state.motor.position[:] + self.motor_encoder_noise,  # actuated joint positions
                self.cassie_state.joint.position[:] + self.joint_encoder_noise,  # unactuated joint positions

                # joint velocities
                self.cassie_state.motor.velocity[:],  # actuated joint velocities
                self.cassie_state.joint.velocity[:]   # unactuated joint velocities
            ])

        # Concatenate robot_state to ext_state
        ext_state = np.concatenate((robot_state, ext_state))

        return ext_state

    def mirror_state(self, state):

        if self.reduced_input:
            index = np.array(
                [0, -1, 2, -3,
                 -4, 5, -6,
                 -12, -13, 14, 15, 16, -7, -8, 9, 10, 11,
                 -22, -23, 24, 25, 26, -17, -18, 19, 20, 21,
                 ])
        else:
            index = np.array(
                [0,
                 1, -2, 3, -4,
                 -10, -11, 12, 13, 14, -5, -6, 7, 8, 9,
                 15, -16, 17,
                 -18, 19, -20,
                 -26, -27, 28, 29, 30, -21, -22, 23, 24, 25,
                 31, -32, 33,
                 37, 38, 39, 34, 35, 36,
                 43, 44, 45, 40, 41, 42])

        if self.clock_based:
            index = np.append(index, np.arange(len(index), len(index) + 3))
        else:
            index = np.append(index, len(index))

        return mirror(state, index)

    @staticmethod
    def mirror_action(action, index=(-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4)):
        return mirror(action, index)

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, self.config)

        self.vis.draw(self.sim)
