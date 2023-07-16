import numpy as np


def reward_engineering_mountain_car(state, action, reward, next_state, done):
    """
    Makes reward engineering to allow faster training in the Mountain Car environment.

    :param state: state.
    :type state: NumPy array with dimension (1, 3).
    :param action: torque.
    :type action: float.
    :param reward: original reward.
    :type reward: float.
    :return: modified reward for faster training.
    :rtype: float.
    """
    # Todo: implement reward engineering
    position = state[0]
    angle_pole = state[2]
    torque = action
    rmodified = -(theta ** 2 + 0.1 * theta_dt ** 2 + 0.001 * torque ** 2)
    return rmodified
