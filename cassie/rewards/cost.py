import numpy as np

from cassie.utils.power_estimation import estimate_power


def kernel(x, coeff=1.):
    return 1 - np.exp(-coeff * x ** 2)


def power_cost(robot_state, robot_mass, qvel, coeff=1e-3, cot=False, debug=False):
    power_estimate, _ = estimate_power(robot_state.motor.torque[:],
                                       robot_state.motor.velocity[:],
                                       positive_only=True)
    if cot:
        power_cost = kernel(power_estimate / (robot_mass * 9.81 * abs(qvel[0])),  coeff)
    else:
        power_cost = kernel(power_estimate, coeff)

    if debug:
        print('[Power Cost: {:.3f}, Power Est: {:.2f}]'.format(power_cost, power_estimate))

    return power_cost
