# gambler's problem from Sutton's book
"""
A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips. If the coin comes up heads, he wins as many dollars as he has staked on that flip; if it is tails, he loses his stake. The game ends when the gambler wins by reaching his goal of $100, or loses by running out of money.
"""

from email import policy
from re import A
import sys
import numpy as np
import matplotlib.pyplot as plt

# implement value iteration for the gambler's problem and solve it for p_h = .25 and p_h .55
"""
The state-value function then gives the probability of winning from each state. A policy is a mapping from levels of capital to stakes. The optimal policy maximizes the probability of reaching the goal. Let p_h denote the probability of the coin coming up heads. If p_h is known, then the entire problem is known and it can be solved, for instance, by value iteration.
"""


def value_iteration(p_h, theta=1e-4, gamma=1.):
    """
    Value iteration for gambler's problem from Sutton's book

    Args:
        p_h (float): Probability of the coin coming up heads
        theta (float, optional): threshold dertermining accuracy of estimation . Defaults to 1e-4.
        gamma (float, optional): discount factor. Defaults to 1..
    """
    
    # reward is zero on all transitions except those on which the gamble reaches his goal,
    rewards = np.zeros(101)
    rewards[100] = 1
    
    # two dummy states corresponding to terminatin with capital of 0 and 100
    V = np.zeros(101)
    
    def one_step_lookahead(s, V, rewards):
        """
        Helper function to calculate the value for all action in a given state

        Args:
            s (int): The gambler's captital
            V (np.array): The vector that contains values at each state
            rewards (int): The reward vector
            
        Returns:
            A vetor conatining the expected value of each action. Its length equals to the number of actions.
        """
        A = np.zeros(101)
        stakes = range(1, min(s, 100-s) + 1)
        print(stakes)
        for a in stakes:
            # rewards[s+a], rewards[s-a] are immediate reward
            # V[s+a], V[s-a] are values of the next states
            # core of the bellman eqation: the expected value of the action is the sum of immediate rewares and the value of the next state
            A[a] = p_h * (rewards[s+a] + V[s+a] * gamma) + (1 - p_h) * (rewards[s-a] + V[s-a] * gamma)
        
        return A
    
    return policy, V

    # value iteration
    while True:
        # stopping condition
        delta = 0
        # update for each state
        for s in range(1, 100):
            # one step lookahead to find the best action
            A = one_step_lookahead(s, V, rewards)
            print(s, A, V)
            best_action_value = np.max(A)
            # calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # update the value function
            V[s] = best_action_value
        if delta < theta:
            break
        
    # create a deterministic policy using the optimal value function
    policy = np.zeros(100)
    for s in range(1, 100):
        # one step lookahead to find the bes action for current state
        A = one_step_lookahead(s, V, rewards)
        best_action = np.max(A)
        policy[s] = best_action
        
    return policy, V