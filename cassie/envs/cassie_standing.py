import gym
import numpy as np

from cassie.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis


# Creating the Standing Environment
class CassieEnv:

    def __init__(self, simrate=60, clock_based=True, state_est=True, reward_cutoff=0.3, target_action_weight=1.0,
                 forces=(0, 0, 0), config="cassie/cassiemujoco/cassie.xml"):

        # Using CassieSim
        self.config = config
        self.sim = CassieSim(self.config)
        self.vis = None

        # Initialize parameters
        self.clock_based = clock_based
        self.state_est = state_est
        self.reward_cutoff = reward_cutoff
        self.target_action_weight = target_action_weight
        self.forces = forces

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
        offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        target = action + (self.target_action_weight * offset)

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

        # Current State
        state = self.get_full_state()

        # Early termination condition
        height_in_bounds = 0.4 < self.sim.qpos()[2] < 3.0

        # Current Reward
        reward = self.compute_reward() if height_in_bounds else 0.0

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

    def compute_reward(self):
        # Cassie weight properties
        gravity = 9.81
        mass = np.sum(self.sim.get_body_mass())
        weight = mass * gravity

        # CoM Position and Velocity
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        # Foot Position
        left_foot_pos = self.cassie_state.leftFoot.position[:]
        right_foot_pos = self.cassie_state.rightFoot.position[:]

        # Add offset to get midfoot position (https://github.com/agilityrobotics/agility-cassie-doc/wiki/Toe-Model)
        midfoot_offset = [0.1762, 0.05219, 0.]
        foot_pos = np.concatenate([left_foot_pos + midfoot_offset, right_foot_pos + midfoot_offset])

        # Foot Force
        foot_grf = self.sim.get_foot_forces()

        # A. Standing Rewards
        # 1. Upper Body Pose Modulation
        pelvis_roll = np.exp(-qpos[4] ** 2)
        pelvis_pitch = np.exp(-qpos[5] ** 2)

        r_pose = 0.5 * pelvis_roll + 0.5 * pelvis_pitch

        # 2. CoM Position Modulation
        # 2a. Horizontal Position Component (target position is the center of the support polygon)
        xy_target_pos = np.array([0.5 * (np.abs(foot_pos[0]) + np.abs(foot_pos[3])),
                                  0.5 * (np.abs(foot_pos[1]) + np.abs(foot_pos[4]))])
        xy_com_pos = np.exp(-(np.linalg.norm(xy_target_pos - qpos[:2])) ** 2)

        # 2b. Vertical Position Component (robot should stand upright and maintain a certain height)
        z_com_pos = np.exp(-(qpos[2] - 0.9) ** 2)

        r_com_pos = 0.5 * xy_com_pos + 0.5 * z_com_pos

        # 3. CoM Velocity Modulation
        # 3a. Horizontal Velocity Component (desired CoM velocity is derived from capture point)
        xy_target_vel = (xy_target_pos - qpos[:2]) / np.sqrt(qpos[2] / gravity)
        xy_com_vel = np.exp(-np.linalg.norm(xy_target_vel - qvel[:2]) ** 2)

        # 3b. Vertical Velocity Component (desired vertical CoM velocity is 0)
        z_com_vel = np.exp(-(qvel[2] ** 2))

        # the reward term for horizontal CoM velocity is deemed invalid and is set to 0
        # when the robot is in the air (no foot contact)
        contact_weight = lambda w: w / (1 + np.exp((weight / 2) - np.linalg.norm(foot_grf)))
        r_com_vel = contact_weight(0.5) * xy_com_vel + 0.5 * z_com_vel

        # 4. Ground Force Modulation
        # Fix: Foot forces are registering as a total of ~500N which doesn't make sense
        # if Cassie's mass is ~33 kg, so instead of having the target_grf be weight / 2,
        # it'll be np.sum(foot_grf) / 2 to signal even distribution still

        target_grf = np.sum(foot_grf) / 2.
        left_grf  = np.exp(-(target_grf - foot_grf[0]) ** 2)
        right_grf = np.exp(-(target_grf - foot_grf[1]) ** 2)

        print(mass, foot_grf, left_grf, right_grf)

        r_grf = 0.5 * left_grf + 0.5 * right_grf

        # Total Reward
        reward = 0.25 * r_pose + 0.25 * r_com_pos + 0.25 * r_com_vel + 0.25 * r_grf

        # B. Standing Cost
        # 5. Ground Contact
        c_contact = 0.3 * np.exp(-np.linalg.norm(foot_grf) ** 2)

        # 6. Power Consumption
        joint_torques = np.array(self.cassie_state.motor.torque[:10])
        joint_velocities = np.array(self.cassie_state.motor.velocity[:10])
        c_power = 0.2 * np.exp(-np.linalg.norm(joint_torques * joint_velocities) ** 2)

        # Total Cost
        cost = c_contact + c_power

        return reward - cost

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
