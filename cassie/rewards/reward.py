import numpy as np
from cassie.utils.quaternion_function import *


def kernel(x, coeff=1.):
    return np.exp(-coeff * x ** 2)


def compute_reward(robot_state, qpos, qvel, fpos, rw=(0.2, 0.2, 0.2, 0.2, 0.2), debug=False):

    # 1. Pelvis Orientation
    target_pel_orient = np.array([1., 0., 0., 0.])
    actual_pel_orient = rotate_to_orient(robot_state.pelvis.orientation[:])
    pel_orient_error  = 1 - np.inner(target_pel_orient, actual_pel_orient) ** 2
    pel_orient_reward = kernel(pel_orient_error, 1000)

    # 2. Pelvis Rotational Velocity
    target_rot_vel = np.zeros(3)
    actual_rot_vel = robot_state.pelvis.rotationalVelocity[:]
    rot_vel_error  = np.linalg.norm(actual_rot_vel - target_rot_vel)
    rot_vel_reward = kernel(rot_vel_error, 1)

    # 3. Pelvis Translational Velocity
    target_trans_vel = np.zeros(3)
    actual_trans_vel = qvel[:3]
    trans_vel_error  = np.linalg.norm(actual_trans_vel - target_trans_vel)
    trans_vel_reward = kernel(trans_vel_error, 1)

    # 4. Pelvis Position
    target_pel_pos = np.array([0.5 * (fpos[0] + fpos[3]),
                               0.5 * (fpos[1] + fpos[4]),
                               0.9])
    actual_pel_pos = qpos[:3]

    pel_pos_error  = np.linalg.norm(actual_pel_pos - target_pel_pos)
    pel_pos_reward = kernel(pel_pos_error, 10)

    # 5. Pelvis/Leg Yaw
    actual_l_foot_orient = quaternion_product(actual_pel_orient, robot_state.leftFoot.orientation)
    actual_r_foot_orient = quaternion_product(actual_pel_orient, robot_state.rightFoot.orientation)

    actual_l_foot_orient_euler = quaternion2euler(actual_l_foot_orient) * [0, 1, 0]  # ROLL PITCH YAW
    actual_r_foot_orient_euler = quaternion2euler(actual_r_foot_orient) * [0, 1, 0]

    target_l_foot_orient = euler2quat(z=actual_l_foot_orient_euler[2], y=actual_l_foot_orient_euler[1], x=actual_l_foot_orient_euler[0])
    target_r_foot_orient = euler2quat(z=actual_r_foot_orient_euler[2], y=actual_r_foot_orient_euler[1], x=actual_r_foot_orient_euler[0])

    l_foot_orient_error = 1 - np.inner(target_l_foot_orient, actual_l_foot_orient) ** 2
    r_foot_orient_error = 1 - np.inner(target_r_foot_orient, actual_r_foot_orient) ** 2

    foot_orient_reward  = 0.5 * kernel(l_foot_orient_error, 1000) + 0.5 * kernel(r_foot_orient_error, 1000)

    if debug:
        print('Rewards: Position [{:.3f}], Orient [PEL: {:.3f}, FOOT: {:.3f}], Velocity [ROT: {:.3f}, TRANS: {:.3f}]'.
              format(pel_pos_reward,
                     pel_orient_reward,
                     foot_orient_reward,
                     rot_vel_reward,
                     trans_vel_reward)
              )

    # compute total reward
    return rw[0] * pel_orient_reward + rw[1] * rot_vel_reward + rw[2] * trans_vel_reward + rw[3] * pel_pos_reward + rw[4] * foot_orient_reward
