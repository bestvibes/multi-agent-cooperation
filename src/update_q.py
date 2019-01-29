import numpy as np
import random
import unittest

from src.util_types import ActionCardinal

class UpdateQ:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, Q_table: dict, state: tuple, action: ActionCardinal, next_state: tuple, reward:int) -> dict:
        max_value = np.amax(list(Q_table[next_state].values()))
        curr_value = Q_table[state][action]
        Q_table[state][action] = curr_value + self.alpha * (reward + self.gamma * max_value - curr_value)

        return Q_table


