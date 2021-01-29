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
                 reward_cutoff=0.3, target_action_weight=1.0, target_height=0.9, training_steps=1e6, learn_command=False,
                 forces=(0, 0, 0), force_fq=100, min_height=0.4, max_height=3.0, max_orient=0, fall_threshold=0.6, encoder_noise=0.2,
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
        self.learn_command = learn_command
        self.debug = debug
        self.writer = writer

        # Encoder noise
        self.encoder_noise = encoder_noise
        self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
        self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)

        # Get simulation properties for dynamics randomization and rewards
        self.mass   = self.sim.get_body_mass()
        self.weight = np.sum(self.mass) * 9.81
        self.friction = self.sim.get_geom_friction()

        # L/R midfoot offset (https://github.com/agilityrobotics/agility-cassie-doc/wiki/Toe-Model)
        # Already included in cassie_mujoco
        # self.midfoot_offset = np.array([0.1762, 0.05219, 0., 0.1762, -0.05219, 0.])
        self.midfoot_offset = np.array([-0.05, 0., 0., -0.05, 0., 0.])
        self.neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646,
                                             -0.6609396704367185, 0.663921021343526])

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

        self.strength_level = 1. / 4.

        self.P = np.array([30., 10., 88., 96., 40.,
                           30., 10., 88., 96., 40., ])
        self.D = np.array([3.0, 1.0, 8.0, 9.6, 4.0,
                           3.0, 1.0, 8.0, 9.6, 4.0, ])

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

        # Set number of actions based on PD gains are learned or not
        # (30 if learning PD gains = 10 jont positions + 2 gains * 10 joints)
        num_actions = 30 if self.learn_PD else 10

        # Action and velocity tracking for large changes
        self.previous_action = np.zeros(num_actions)
        self.previous_velocity = self.cassie_state.motor.velocity[:]
        self.previous_acceleration = np.zeros(len(self.previous_velocity))

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

        # TODO: Add noise to gains either here or in reset fn
        if self.learn_PD:
            target_P = self.P + (action[10:20] * self.P)
            target_D = self.D + (action[20:]   * self.D)
            action   = action[:10]

        # Create Target Action
        target = (1.25 * action + self.offset) - self.motor_encoder_noise

        foot_pos = np.zeros(6)
        self.sim.foot_pos(foot_pos)
        prev_foot = deepcopy(foot_pos)
        self.u = pd_in_t()

        # Apply Action
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]     if not self.learn_PD else target_P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i + 5] if not self.learn_PD else target_P[i + 5]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]     if not self.learn_PD else target_D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i + 5] if not self.learn_PD else target_D[i + 5]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        # Send action input (u) into sim and update cassie_state
        self.cassie_state = self.sim.step_pd(self.u)

        # Update tracking variables
        self.sim.foot_pos(foot_pos)
        self.l_foot_vel = (foot_pos[0:3] - prev_foot[0:3]) / 0.0005
        self.r_foot_vel = (foot_pos[3:6] - prev_foot[3:6]) / 0.0005

    def step(self, action):

        # TODO: hold pelvis in place
        # self.sim.hold()

        # TODO: release pelvis
        # self.sim.release()

        if self.timestep % random.randint(1, self.force_fq) == 0:
            if np.sum(self.target_speed) == 0 and np.all(np.abs(self.sim.qvel()[:2]) < 0.2):
                # Apply perturbations to the pelvis in random directions
                self.sim.apply_force([random.uniform(-self.forces[0], self.forces[0]),
                                      random.uniform(-self.forces[1], self.forces[1]),
                                      random.uniform(-self.forces[2], self.forces[2]),
                                      0,
                                      0])
            else:
                force_dir = np.sign(self.target_speed)

                # Apply perturbations to the pelvis in the direction of the pelvis
                self.sim.apply_force([force_dir[0] * self.forces[0],
                                      force_dir[1] * self.forces[1],
                                      force_dir[2] * self.forces[2],
                                      0,
                                      0])

        # random changes to target yaw of the pelvis
        if np.random.rand() < 0.1 and self.learn_command:
            self.target_orientation += np.random.uniform(-self.max_orient, self.max_orient, size=3)

        # simulating delays
        simrate = self.simrate + np.random.randint(-5, 5) if not self.test else self.simrate

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
        alive_bounds = self.min_height < self.sim.qpos()[2] < self.max_height

        reward = self.compute_reward(qpos, qvel, foot_pos) - self.compute_cost(action, foot_grf, qvel) \
            if alive_bounds else -self.timestep

        # Done Condition
        done = not alive_bounds or reward < self.reward_cutoff

        # Update timestep counter
        self.timestep += 1
        self.total_steps += 1

        return state, reward, done, {}

    def reset(self, reset_ratio=0, use_phase=False):
        # update strength
        self.strength_level = self.strength_level + (8e-8 * self.timestep) if self.strength_level < 1 / 2 and not self.test else 1 / 2

        # reset variables
        self.timestep = 0
        self.sim.full_reset()
        self.reset_cassie_state()
        self.previous_action = np.zeros(self.action_space.shape[0])
        self.previous_velocity = self.cassie_state.motor.velocity[:]
        self.previous_acceleration = np.zeros(len(self.previous_velocity))

        self.P = np.array([30., 10., 88., 96., 40.,
                           30., 10., 88., 96., 40., ]) * self.strength_level
        self.D = np.array([3.0, 1.0, 8.0, 9.6, 4.0,
                           3.0, 1.0, 8.0, 9.6, 4.0, ]) * self.strength_level

        # reset target variables
        self.target_orientation = np.zeros(3)
        if self.learn_command:
            self.target_speed[0] = random.uniform(self.min_speed[0], self.max_speed[0])
            self.target_speed[1] = random.uniform(self.min_speed[0], self.max_speed[0])
            self.target_speed[2] = random.uniform(self.min_speed[0], self.max_speed[0])
        else:
            self.target_speed = np.zeros(3)

        if not self.test:
            # TODO: randomize mass and recalculate weight
            self.sim.set_body_mass(randomize_mass(self.mass, low=0.9, high=1.1))
            self.weight = np.sum(self.sim.get_body_mass()) * 9.81

            # TODO: randomize ground friction
            self.sim.set_geom_friction(randomize_friction(self.friction, low=0.9, high=1.1))

            # TODO: joint noise error
            self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
            self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)

        else:
            self.motor_encoder_noise = np.zeros(10)
            self.joint_encoder_noise = np.zeros(6)

        # only randomize init height for standing (commanded velocities are all zero)
        if np.sum(self.target_speed) == 0 and np.sum(self.target_orientation) == 0:
            # TODO: randomize initial height
            self.randomize_init_height()

        # reset with perturbations and/or from a different stance (use_phase=True)
        if np.random.rand() < reset_ratio:

            # initialize qvel and get desired xyz velocities
            qvel = np.copy(self.sim.qvel())
            x_speed = random.uniform(self.min_speed[0], self.max_speed[0])
            y_speed = random.uniform(self.min_speed[0], self.max_speed[0])
            z_speed = random.uniform(self.min_speed[0], self.max_speed[0])

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

        init_height = random.choice([mid_start, high_start])

        self.sim.set_qpos(init_height)

    def compute_reward(self, qpos, qvel, foot_pos, rw=(0.1, 0.1, 0.3, 0.1, 0.1, 0.3)):
        # rw=(pose, CoM pos, CoM vel, foot placement, foot/pelvis orientation, GRF)

        left_foot_pos  = foot_pos[:3]
        right_foot_pos = foot_pos[3:]

        # midfoot position
        foot_pos = np.concatenate([left_foot_pos - self.midfoot_offset[:3], right_foot_pos - self.midfoot_offset[3:]])

        # 1. Pelvis Orientation (Commanded Yaw or Gaze (roll, pitch, and yaw) )
        pelvis_orient_coeff = 1e4

        # convert target orientation to quaternion
        target_orient = euler2quat(z=self.target_orientation[2], y=self.target_orientation[1], x=self.target_orientation[0])

        # calculate orientation error
        orient_error = 1 - np.inner(qpos[3:7], target_orient) ** 2

        r_pose = np.exp(-pelvis_orient_coeff * orient_error ** 2)

        # 2. CoM Position Modulation
        com_pos_coeff = [250, 500, 100]

        xy_target_pos = np.array([0.5 * (foot_pos[0] + foot_pos[3]),
                                  0.5 * (foot_pos[1] + foot_pos[4])])

        x_com_pos = np.exp(-com_pos_coeff[0] * np.linalg.norm(qpos[0] - xy_target_pos[0]) ** 2)
        y_com_pos = np.exp(-com_pos_coeff[1] * np.linalg.norm(qpos[1] - xy_target_pos[1]) ** 2)
        z_com_pos = np.exp(-com_pos_coeff[2] * (qpos[2] - self.target_height) ** 2)

        r_com_pos = 0.5 * y_com_pos + 0.3 * x_com_pos + 0.2 * z_com_pos

        # 3. CoM Velocity Modulation (Commanded XY)
        x_target_vel = np.exp(-100 * np.linalg.norm(qvel[0] - self.target_speed[0]) ** 2)
        y_target_vel = np.exp(-50  * np.linalg.norm(qvel[1] - self.target_speed[1]) ** 2)
        z_target_vel = np.exp(-25  * np.linalg.norm(qvel[2] - self.target_speed[2]) ** 2)

        r_com_vel = 0.75 * x_target_vel + 0.2 * y_target_vel + 0.05 * z_target_vel

        # 4. Foot Placement
        foot_placement_coeff = 1e3

        # 4a. Foot Alignment
        r_feet_align = np.exp(-foot_placement_coeff * (foot_pos[0] - foot_pos[3]) ** 2)

        # 4b. Feet Width
        target_width = 0.18  # m = 18 cm makes the support polygon a square
        width_thresh = 0.02  # m = 2 cm
        feet_width = np.linalg.norm([foot_pos[1], foot_pos[4]])

        if feet_width < target_width - width_thresh:
            r_foot_width = np.exp(-foot_placement_coeff * (feet_width - (target_width - width_thresh)) ** 2)
        elif feet_width > target_width + width_thresh:
            r_foot_width = np.exp(-foot_placement_coeff * (feet_width - (target_width + width_thresh)) ** 2)
        else:
            r_foot_width = 1.

        # 4c. Foot Orientation (keep the feet neutral)
        r_foot_orient_error = 1 - np.inner(self.neutral_foot_orient, self.sim.xquat("left-foot"))  ** 2
        l_foot_orient_error = 1 - np.inner(self.neutral_foot_orient, self.sim.xquat("right-foot")) ** 2
        r_foot_neutral = np.exp(-1e4 * np.linalg.norm([l_foot_orient_error, r_foot_orient_error]) ** 2)

        r_foot_placement = 0.6 * r_foot_width + 0.3 * r_foot_neutral + 0.1 * r_feet_align\
            if np.sum(self.target_speed) == 0 else 0.6 * r_foot_width + 0.4 * r_foot_neutral

        # 5. Foot/Pelvis Orientation TODO: Fix for gazing, feet needs to be detached from pelvis
        fp_orientation_coeff = 100
        foot_yaw = np.array([qpos[8], qpos[22]])

        # convert quaternion values to euler [roll, pitch, yaw]
        pelvis_orient = quaternion2euler(qpos[3:7])

        left_foot_orient  = np.exp(-fp_orientation_coeff * (foot_yaw[0] - pelvis_orient[2]) ** 2)
        right_foot_orient = np.exp(-fp_orientation_coeff * (foot_yaw[1] - pelvis_orient[2]) ** 2)

        r_fp_orient = 0.5 * left_foot_orient + 0.5 * right_foot_orient

        # 6. Distance covered
        target_dist = (self.target_speed[0] * 0.03) * self.timestep
        r_dist  = np.exp(-50 * (target_dist - qpos[0]) ** 2)

        # Total Reward
        reward = (rw[0] * r_pose
                  + rw[1] * r_com_pos
                  + rw[2] * r_com_vel
                  + rw[3] * r_foot_placement
                  + rw[4] * r_fp_orient
                  + rw[5] * r_dist)

        if self.writer is not None and self.debug and not self.test:
            # log episode reward to tensorboard
            self.writer.add_scalar('env_reward/pose', r_pose, self.total_steps)
            self.writer.add_scalar('env_reward/com_pos', r_com_pos, self.total_steps)
            self.writer.add_scalar('env_reward/com_vel', r_com_vel, self.total_steps)
            self.writer.add_scalar('env_reward/foot_placement', r_foot_placement, self.total_steps)
            self.writer.add_scalar('env_reward/foot_orientation', r_fp_orient, self.total_steps)
        elif self.writer is None and self.debug:
            print('[{:3}] Rewards: Pose [{:.3f}], CoM [{:.3f}, {:.3f}], Foot [{:.3f}, {:.3f}, {:.3f}], DIST[{:.3f}]]'.format(self.timestep,
                                                                                                                            r_pose,
                                                                                                                            r_com_pos,
                                                                                                                            r_com_vel,
                                                                                                                            r_foot_placement,
                                                                                                                            r_fp_orient,
                                                                                                                            r_foot_neutral,
                                                                                                                            r_dist,
                                                                                                                           ))

        return reward

    def compute_cost(self, action, foot_frc, qvel, cw=(0.1, 0., 0., 0., 0.5)):
        # cost_coeff = self.total_steps / self.training_steps if not self.test else 1.

        # 1. Power Consumption (Torque and Velocity) and Cost of Transport (CoT = P / [M * v])
        power_estimate, power_info = estimate_power(self.cassie_state.motor.torque[:10],
                                                    self.cassie_state.motor.velocity[:10],
                                                    positive_only=True)

        # calculate cost of transport in the XY
        # cot = power_estimate / (np.sum(self.mass) * abs(qvel[0])) if abs(qvel[0]) > 0 else power_estimate

        c_power = 1. - np.exp(-1e-4 * power_estimate ** 2)

        # 2. Action Cost
        action_diff = np.subtract(self.previous_action[:10], action[:10])
        c_action = 1 - np.exp(-250 * np.sum(action_diff) ** 2)

        # 3. Motor Jerk Cost (Proportional Motor Vel to Pelvis Vel)
        motor_accel = np.subtract(self.previous_velocity, self.cassie_state.motor.velocity[:])
        motor_jerk = np.subtract(self.previous_acceleration, motor_accel)
        c_mjerk = 1 - np.exp(-5 * np.linalg.norm(motor_jerk) ** 2)

        # 4. Foot Dragging (Lateral)
        ML_forces = 1 - np.exp(-1e-2 * np.linalg.norm([foot_frc[1], foot_frc[4]]) ** 2)
        AP_forces = 1 - np.exp(-1e-2 * np.linalg.norm([foot_frc[0], foot_frc[3]]) ** 2)

        c_drag = 0.6 * ML_forces + 0.4 * AP_forces if np.sum(qvel[:2]) < 0.1 else 0

        # TODO: Reward or cost that places foot with the least GRFz or the foot in the air
        #  in the same direction of the XY pelvis velocity (Capture Point)

        # 5. Contact Cost (Keep at least one foot on the ground)
        c_contact = 1 if (foot_frc[2] + foot_frc[5]) <= 0 else 0

        # Total Cost
        cost = cw[0] * c_power + cw[1] * c_action + cw[2] * c_mjerk + cw[3] * c_drag + cw[4] * c_contact

        # Update previous variables
        self.previous_action = action
        self.previous_acceleration = motor_accel
        self.previous_velocity = self.cassie_state.motor.velocity[:]

        if self.writer is not None and self.debug and not self.test:
            # log episode reward to tensorboard
            self.writer.add_scalar('env_cost/action_change', c_action, self.total_steps)
            self.writer.add_scalar('env_cost/c_mjerk', c_mjerk, self.total_steps)
            self.writer.add_scalar('env_cost/power', c_power, self.total_steps)
            self.writer.add_scalar('env_cost/drag', c_drag, self.total_steps)
            self.writer.add_scalar('env_cost/contact', c_contact, self.total_steps)
        elif self.writer is None and self.debug:
            print('Costs:\t Action Change [{:.3f}], Motor Jerk [{:.3f}], Power [{:.3f}], Drag [{:.3f}], '
                  'Foot Contact[{:.3f}]\n'.format(c_action, c_mjerk, c_power, c_drag, c_contact))

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

        # command consists of [forward velocity, lateral velocity, yaw/gaze]
        # command = np.concatenate((self.target_speed[:2], self.target_orientation))  # turn on for gaze
        command = np.concatenate((self.target_speed[:2], self.target_orientation[2:]))

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
