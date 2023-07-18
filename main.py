import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from ql_agent import QLearningAgent

os.environ['KMP_DUPLICATE_LIB_OK']='True'

NUM_EPISODES = 20  # Number of episodes used for training
NUM_SHOW = 100 # Number of episodes used for show trained


# Initiating the CartPole environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Creating the QL agent
# Here define the parameters for state discretization
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -4
cartVelocityMax = 4
poleAngleVelocityMin = -5
poleAngleVelocityMax = 5
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]
agent = QLearningAgent(numberOfBins, action_size, 0.2, 0.1, 1, lowerBounds, upperBounds)

sumRewards = []
for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state, _ = env.reset()
    state = list(state)
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = []
    print("Simulating episode {}".format(episodes))
    done = False
    while not done:
        state_index = agent.get_state_index(state)
        action = agent.epsilon_greedy_action(state,episodes)
        # Take action, observe reward and new state
        next_state, reward, done, _, _ = env.step(action)
        next_state = list(next_state)
        next_state_index = agent.get_state_index(next_state)
        agent.learn(state_index,action,reward,next_state_index,done)
        cumulative_reward.append(reward)
    print("Sum of rewards {}".format(np.sum(cumulative_reward)))
    sumRewards.append(np.sum(cumulative_reward))      
plt.plot(sumRewards, color='blue',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.yscale('log')
plt.savefig('agent_training.png')
plt.show()


#Best Policy
# env_train = gym.make("CartPole-v1")
return_history = []
env_train = gym.make("CartPole-v1", render_mode ="human")
state, _ = env_train.reset()
env_train.render()
for episodes in range(1, NUM_SHOW + 1):
    print(episodes)
    done = False
    state, _ = env_train.reset()
    cumulative_reward = 0.0
    while not done:
        action = agent.get_right_action(state)
        state, reward, done, _, _ = env_train.step(action)
        cumulative_reward+= reward
    return_history.append(cumulative_reward)
#mean of rewards of agent already trained
print("Mean return: ", np.mean(return_history))

plt.plot(return_history, color='blue',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards in Episode')
plt.yscale('log')
plt.savefig('evaluate.png')
plt.show()
