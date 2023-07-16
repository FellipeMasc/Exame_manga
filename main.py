import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from ql_agent import QLearningAgent
from utils import reward_engineering_mountain_car
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK']='True'

NUM_EPISODES = 300  # Number of episodes used for training
RENDER = True  # If the Mountain Car environment should be rendered

# Comment this line to enable training using your GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.compat.v1.disable_eager_execution()

# Initiating the CartPole environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Creating the QL agent
# Here define the parameters for state discretization
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]
agent = QLearningAgent(numberOfBins, action_size, 0.5, 0.1, 1, lowerBounds, upperBounds)

return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state = env.reset()
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    for time in range(1, 500):
        if RENDER:
            env.render()  # Render the environment for visualization
        # Select action
        action = agent.get_greedy_action(state)
        # Take action, observe reward and new state
        next_state, reward, done, _ = env.step(action)

        
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
    return_history.append(cumulative_reward)
    # Every 10 episodes, update the plot for training monitoring
    if episodes % 20 == 0:
        plt.plot(return_history, 'b')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.show(block=False)
        plt.pause(0.1)
        plt.savefig('dqn_training.' + fig_format, format="png")
plt.pause(1.0)