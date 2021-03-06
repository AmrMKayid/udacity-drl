import random
import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.eps = 0.005
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # return np.random.choice(self.nA)
        if random.random() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.Q[state][action] += 1
        self.Q[state][action] = self.update_Q_expected_sarsa(
            state, action, reward, next_state)
        if done:
            self.eps /= 2.0

    def update_Q_expected_sarsa(self, state, action, reward, next_state, alpha=0.05, gamma=.9):

        current = self.Q[state][action]
        policy_s = np.ones(self.nA) * self.eps / self.nA
        policy_s[np.argmax(self.Q[next_state])] = 1 - \
            self.eps + (self.eps / self.nA)
        Qsa_next = np.dot(self.Q[next_state], policy_s)
        target = reward + (gamma * Qsa_next)
        new_value = current + (alpha * (target - current))
        return new_value
