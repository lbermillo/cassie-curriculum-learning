import gym
import numpy as np

from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis


# Creating the Standing Environment
class CassieEnv:

    def __init__(self, simrate=60, clock_based=True, state_est=True, reward_cutoff=0.3, target_action_weight=1.0,
                 forces=(0, 0, 0), min_height=0.4, max_height=3.0, config="cassie/cassiemujoco/cassie.xml"):

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

        # Cassie properties
        self.mass = np.sum(self.sim.get_body_mass())
        self.weight = self.mass * 9.81

        # L/R midfoot offset (https://github.com/agilityrobotics/agility-cassie-doc/wiki/Toe-Model)
        self.midfoot_offset = np.array([0.1762, 0.05219, 0., 0.1762, -0.05219, 0.])

        # needed to calculate accelerations
        self.prev_velocity = np.copy(self.sim.qvel())[:3]

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

    def reset(self):
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

    def compute_reward(self, qpos, qvel, foot_pos, foot_grf, rw=(0.15, 0.1, 0.1, 0.3, 0.2, 0.15), multiplier=5.):
        left_foot_pos = foot_pos[:3]
        right_foot_pos = foot_pos[3:]

        # midfoot position
        foot_pos = np.concatenate([left_foot_pos - self.midfoot_offset[:3], right_foot_pos - self.midfoot_offset[3:]])

        # A. Standing Rewards
        # 1. Pelvis Orientation
        r_pelvis_roll = np.exp(-qpos[4] ** 2)
        r_pelvis_pitch = np.exp(-qpos[5] ** 2)
        r_pelvis_yaw = np.exp(-qpos[6] ** 2)

        r_pose = 0.335 * r_pelvis_roll + 0.33 * r_pelvis_pitch + 0.335 * r_pelvis_yaw

        # 2. CoM Position Modulation
        # 2a. Horizontal Position Component (target position is the center of the support polygon)
        xy_target_pos = np.array([0.5 * (np.abs(foot_pos[0]) + np.abs(foot_pos[3])),
                                  0.5 * (np.abs(foot_pos[1]) + np.abs(foot_pos[4]))])
        xy_com_pos = np.exp(-np.sum(qpos[:2] - xy_target_pos) ** 2)

        # 2b. Vertical Position Component (robot should stand upright and maintain a certain height)
        z_target_pos = 0.9
        z_com_pos = np.exp(-(qpos[2] - z_target_pos) ** 4)

        r_com_pos = 0.5 * xy_com_pos + 0.5 * z_com_pos

        # 3. CoM Velocity Modulation
        r_com_vel = np.exp(-multiplier * np.sum(qvel[:3]) ** 2)

        # 4. Foot Placement
        # 4a. Foot Alignment
        feet_x_pos = np.array([foot_pos[0], foot_pos[3]])
        r_feet_align = np.exp(-np.sum(np.abs(feet_x_pos)) ** 2)

        # 4b. Feet Width
        target_width = 0.2695434287408531
        feet_width = np.abs(foot_pos[1]) + np.abs(foot_pos[4])
        r_foot_width = np.exp(-np.sum(feet_width - target_width) ** 2)

        r_foot_placement = 0.5 * r_feet_align + 0.5 * r_foot_width

        # 5. Foot/Pelvis Orientation
        foot_yaw = np.array([qpos[8], qpos[22]])
        left_foot_orient = np.exp(-np.sum(foot_yaw[0] - qpos[6]) ** 2)
        right_foot_orient = np.exp(-np.sum(foot_yaw[1] - qpos[6]) ** 2)

        r_fp_orient = 0.5 * left_foot_orient + 0.5 * right_foot_orient

        # 7. Ground Force Modulation (Even Vertical Foot Force Distribution)
        grf_tolerance = 10

        # GRF target discourages shear forces and incites even vertical foot force distribution
        target_grf = np.array([0., 0., np.sum(foot_grf) / 2.])
        left_grf = np.exp(-(np.sum(foot_grf[:3] - target_grf) / grf_tolerance) ** 2)
        right_grf = np.exp(-(np.sum(foot_grf[3:] - target_grf) / grf_tolerance) ** 2)

        # reward is only activated when both feet are down
        r_grf = 0.5 * left_grf + 0.5 * right_grf if foot_pos[2] < 2e-3 and foot_pos[5] < 2e-3 else 0.

        # Initial qpos for reference
        # Pelvis Position
        # 0.0, 0.0, 1.01,

        # Pelvis Orientation
        # 1.0, 0.0, 0.0, 0.0,

        # Left Leg
        # 0.0045, 0.0, 0.4973,
        # 0.9784830934748516, -0.016399716640763992, 0.017869691242100763, -0.2048964597373501,
        # -1.1997, 0.0, 1.4267, 0.0, -1.5244, 1.5244, -1.5968,

        # Right Leg
        # -0.0045, 0.0, 0.4973,
        # 0.978614127766972, 0.0038600557257107214, -0.01524022001550036, -0.20510296096975877,
        # -1.1997, 0.0, 1.4267, 0.0, -1.5244, 1.5244, -1.5968

        # Total Reward
        reward = (rw[0] * r_pose
                  + rw[1] * r_com_pos
                  + rw[2] * r_com_vel
                  + rw[3] * r_foot_placement
                  + rw[4] * r_fp_orient
                  + rw[5] * r_grf)

        return reward

    def compute_cost(self, qpos, foot_pos, foot_grf, cw=(0.3, 0.1, 0.2, 0.4)):
        # 1. Ground Contact
        c_contact = 1. if np.sum(foot_grf[2], foot_grf[5]) == 0. else 0.

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

        power_threshold = 110  # Watts (Positive Work only)
        c_power = 1. / (1. + np.exp(-(power_estimate - power_threshold)))

        # 3. Foot Dragging
        left_drag_cost  = np.exp(-foot_pos[2] ** 2) if foot_grf[2] > 0. and np.sum(foot_grf[:2])  != 0. else 0.
        right_drag_cost = np.exp(-foot_pos[5] ** 2) if foot_grf[5] > 0. and np.sum(foot_grf[3:5]) != 0. else 0.

        c_foot_drag = 0.5 * left_drag_cost + 0.5 * right_drag_cost

        # 4. Falling
        c_fall = 1 if qpos[2] < self.min_height else 0

        # Total Cost
        cost = cw[0] * c_contact + cw[1] * c_power + cw[2] * c_foot_drag + cw[3] * c_fall

        return cost

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

            # Concatenate robot_state to ext_state
            ext_state = np.concatenate((robot_state, ext_state))

        return ext_state

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, self.config)

        self.vis.draw(self.sim)
