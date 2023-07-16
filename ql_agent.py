import numpy as np


# def compute_greedy_policy_as_table(q):
#     """
#     Computes the greedy policy as a table.
#
#     :param q: action-value table.
#     :type q: bidimensional numpy array.
#     :return: greedy policy table.
#     :rtype: bidimensional numpy array.
#     """
#     policy = np.zeros(q.shape)
#     for s in range(q.shape[0]):
#         policy[s, greedy_action(q, s)] = 1.0
#     return policy


def epsilon_greedy_action(q, state, epsilon):
    """
    Computes the epsilon-greedy action.

    :param q: action-value table.
    :type q: bidimensional numpy array.
    :param state: current state.
    :type state: int.
    :param epsilon: probability of selecting a random action.
    :type epsilon: float.
    :return: epsilon-greedy action.
    :rtype: int.
    """
    rand_number = np.random.rand()
    if rand_number > epsilon:
        return greedy_action(q,state)
    else:
        rand_action = np.random.randint(0, len(q[state, :]))
        return rand_action


def greedy_action(q, state):
    """
    Computes the greedy action.

    :param q: action-value table.
    :type q: bidimensional numpy array.
    :param state: current state.
    :type state: Numpy array (1,4).
    :return: greedy action.
    :rtype: int.
    """
    return np.argmax(q[state, :])


class QLearningAgent:
    """
    Represents a model-free reinforcement learning algorithm.
    """
    def __init__(self, num_gaps, num_actions, epsilon, alpha, gamma, num_lower, num_upper):
        """
        Creates a model-free reinforcement learning algorithm.

        :param num_gaps: number of gaps in each feature.
        :type num_gaps: Numpy array (1,4).
        :param num_actions: number of actions of the cart.
        :type num_actions: int.
        :param epsilon: probability of selecting a random action in epsilon-greedy policy.
        :type epsilon: float.
        :param alpha: learning rate.
        :type alpha: float.
        :param gamma: discount factor.
        :type gamma: float.
        :param num_lower: lower bound for each feature
        :type num_lower: Numpy array (1,4)
        :param num_upper: upper bound for each feature
        :type num_upper: Numpy array (1,4)
        """
        self.action_number = num_actions
        self.num_gaps = num_gaps
        self.lower = num_lower
        self.upper = num_upper
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q = np.random.uniform(low=0, high=1, size=(num_gaps[0], num_gaps[1], num_gaps[2], num_gaps[3], self.action_number))

    def get_state_index(self,state):
        """
        Returns the index of the features already discretized
        """
        cart_position_gap = np.linspace(self.lower[0], self.upper[0], self.num_gaps[0])
        cart_velocity_gap = np.linspace(self.lower[1], self.upper[1], self.num_gaps[1])
        pole_angle_gap = np.linspace(self.lower[2], self.upper[2], self.num_gaps[2])
        pole_angular_velocity_gap = np.linspace(self.lower[3], self.upper[3], self.num_gaps[3])

        index_position = np.maximum(np.digitize(state[0], cart_position_gap) - 1)
        index_velocity = np.maximum(np.digitize(state[1], cart_velocity_gap) - 1)
        index_angle = np.maximum(np.digitize(state[2], pole_angle_gap) - 1)
        index_angle_velocity = np.maximum(np.digitize(state[3], pole_angular_velocity_gap) - 1)

        return index_position, index_velocity, index_angle, index_angle_velocity

    def update_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.
        """
        self.epsilon *= 0.99
        if self.epsilon < 0.01:
            self.epsilon = 0.01

    def get_greedy_action(self, state):
        """
        Returns a greedy action considering the policy of the RL algorithm.

        :param state: current state.
        :type state: int.
        :return: greedy action considering the policy of the RL algorithm.
        :rtype: int.
        """
        return epsilon_greedy_action(self.q,state,self.epsilon)

    def learn(self, state, action, reward, next_state):
        value_max = np.max(self.q[next_state, :])
        self.q[state, action] = self.q[state, action] + self.alpha * (
                    reward + self.gamma * value_max - self.q[state, action])


