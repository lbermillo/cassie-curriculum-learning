import numpy as np


def mirror(arr, index):
    # get signs from the swap indices
    signs = np.sign(index)

    # return mirrored array
    return signs * np.array([arr[int(np.abs(i))] for i in index])


def mirror_action(action, index=(-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4)):
    return mirror(action, index)


def mirror_state(state, reduced_input=False, clock=True):
    if reduced_input:
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

    if clock:
        index = np.append(index, np.arange(len(index), len(index) + 3))
    else:
        index = np.append(index, len(index))

    return mirror(state, index)


# Implementation based on Learning Symmetric and Low-Energy Locomotion [https://arxiv.org/pdf/1801.08093.pdf]
def compute_mirror_loss(states, policy, env):
    """This is only for reference"""

    # initialize loss
    mirror_loss = 0

    for state in states:
        # convert state to numpy from tensor
        state = state.cpu().data.numpy()

        # get policy output
        action = policy.act(state)

        # mirror state
        mirrored_state = env.mirror_state(state)

        # mirror action
        mirrored_action = env.mirror_action(policy.act(mirrored_state))

        # compute mirror loss
        mirror_loss += np.linalg.norm(action - mirrored_action) ** 2

    return mirror_loss / len(states)
