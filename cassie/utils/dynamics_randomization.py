import numpy as np


def randomize_mass(mass, low, high):
    pelvis_mass_range = [[low * mass[1], high * mass[1]]]  # 1
    hip_mass_range    = [[low * mass[2], high * mass[2]],  # 2->4 and 14->16
                         [low * mass[3], high * mass[3]],
                         [low * mass[4], high * mass[4]]]

    achilles_mass_range    = [[low * mass[5],   high * mass[5]]]  # 5 and 17
    knee_mass_range        = [[low * mass[6],   high * mass[6]]]  # 6 and 18
    knee_spring_mass_range = [[low * mass[7],   high * mass[7]]]  # 7 and 19
    shin_mass_range        = [[low * mass[8],   high * mass[8]]]  # 8 and 20
    tarsus_mass_range      = [[low * mass[9],   high * mass[9]]]  # 9 and 21
    heel_spring_mass_range = [[low * mass[10], high * mass[10]]]  # 10 and 22
    fcrank_mass_range      = [[low * mass[11], high * mass[11]]]  # 11 and 23
    prod_mass_range        = [[low * mass[12], high * mass[12]]]  # 12 and 24
    foot_mass_range        = [[low * mass[13], high * mass[13]]]  # 13 and 25

    side_mass = hip_mass_range + achilles_mass_range \
                + knee_mass_range + knee_spring_mass_range \
                + shin_mass_range + tarsus_mass_range \
                + heel_spring_mass_range + fcrank_mass_range \
                + prod_mass_range + foot_mass_range

    mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
    mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

    return np.clip(mass_noise, a_min=0, a_max=None)


def randomize_friction(friction, low, high):
    friction_noise = []
    translational = np.random.uniform(low, high)
    torsional = np.random.uniform(1e-4, 5e-4)
    rolling = np.random.uniform(1e-4, 2e-4)
    for _ in range(int(len(friction) / 3)):
        friction_noise += [translational, torsional, rolling]

    return np.clip(friction_noise, a_min=0, a_max=None)
