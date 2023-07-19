import numpy as np

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

    def epsilon_greedy_action(self,state,index):
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
        if index < 500:
            return np.random.choice(self.action_number)   
        
        rand_number = np.random.random()
        if index > 7000:
            self.epsilon = 0.999 * self.epsilon
            
        if rand_number < self.epsilon:
            return np.random.choice(self.action_number)
        else:
            # return np.random.choice(np.where(self.q[self.get_state_index(state)]==np.max(self.q[self.get_state_index(state)]))[0])
            state_index = self.get_state_index(state)
            return np.argmax(self.q[state_index[0], state_index[1], state_index[2], state_index[3], :])

    def get_state_index(self,state):
        """
        Returns the index of the features already discretized
        """
        cart_position_gap = np.linspace(self.lower[0], self.upper[0], self.num_gaps[0])
        cart_velocity_gap = np.linspace(self.lower[1], self.upper[1], self.num_gaps[1])
        pole_angle_gap = np.linspace(self.lower[2], self.upper[2], self.num_gaps[2])
        pole_angular_velocity_gap = np.linspace(self.lower[3], self.upper[3], self.num_gaps[3])

        index_position = np.maximum(np.digitize(state[0], cart_position_gap) - 1,0)
        index_velocity = np.maximum(np.digitize(state[1], cart_velocity_gap) - 1,0)
        index_angle = np.maximum(np.digitize(state[2], pole_angle_gap) - 1,0)
        index_angle_velocity = np.maximum(np.digitize(state[3], pole_angular_velocity_gap) - 1,0)

        return tuple([index_position,index_velocity,index_angle,index_angle_velocity])   

    def learn(self, state, action, reward, next_state,done):
        q_max_next = np.max(self.q[next_state])
        if not done:
            error=reward+self.gamma*q_max_next-self.q[state+(action,)]
            self.q[state+(action,)] = self.q[state+(action,)]+ self.alpha * error
        else:
            error=reward-self.gamma*q_max_next-self.q[state+(action,)]
            self.q[state+(action,)] = self.q[state+(action,)] + self.alpha * error

    def get_right_action(self,state):
        state_index = self.get_state_index(state)
        return np.argmax(self.q[state_index[0], state_index[1], state_index[2], state_index[3], :])
