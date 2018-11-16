import numpy as np
import random
import unittest

class UpdateQ:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, Q_table, state, action, next_state, reward):
        max_value = np.max([Q_table[next_state][a] for a in Q_table[next_state]])
        curr_value = Q_table[state][action]
        Q_table[state][action] = curr_value + self.alpha * (reward + self.gamma * max_value - curr_value)

        return Q_table


