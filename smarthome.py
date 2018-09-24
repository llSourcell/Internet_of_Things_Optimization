#plot results
import matplotlib
#matrix math
import numpy as np
#Smart Home Environment
from lib.envs.blackjack import SmartHomeEnv

#initialize environment
env = SmartHomeEnv()

def update(dict_sum, dict_count, tra, discount_factor):
    """
    Cauculate the reward of a given trajectory and record them in dict_sum, dict_count.
    Args:
        dict_sum: the dictionary that record the sum of reward of different states.
        dict_count: the dictionary that record the count of different states.
        tra: trajectory.
    Returns:
        dict_sum: updated
        dict_count: updated
    """
    value = 0.0
    for t in tra[::-1]:
        observation, reward, _ = t
        value = value * discount_factor + reward
        dict_sum[observation] += value
        dict_count[observation] += 1
    return dict_sum, dict_count


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function for a given policy using sampling.
    Args:
        policy: A function that maps an observation to action probabilities.
        env: Smart Home environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    #init, observation: (Cooling Demands, electricity prices, electricity consumption)
    observation = env.reset()
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    memory = []
    for i in range(num_episodes):
        #take an action
        action = policy(observation)
        observation, reward, done, _ = env.step(action)
        memory.append((observation, reward, done))
        if done:
            observation = env.reset()
            returns_sum, returns_count = update(
                returns_sum,
                returns_count,
                memory,
                discount_factor=discount_factor)
            memory = []
    for i in returns_sum.keys():
        V[i] = returns_sum[i] / returns_count[i]
    return V


def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

#plot results
V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title='10,000 Steps')

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title='500,000 Steps')
