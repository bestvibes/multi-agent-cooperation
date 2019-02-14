import random

from src.util_types import ActionCardinal
from src.algorithm import Algorithm

class DummyCardinalStationaryPolicy(object):
    def __call__(self, state):
        return ActionCardinal.STAY

class DummyCardinalRandomPolicy(object):
    def __call__(self, state):
        return ActionCardinal(random.randrange(0, len(ActionCardinal)))

class TrainDummyAgent(Algorithm):
    def __init__(self, dummy_policy):
        self.policy = dummy_policy

    def select_action(self, state):
        return self.policy(state)

    def train_episode_step(self, state, action, next_state, reward):
        pass

    def get_policy(self):
        return self.policy

def zero_reward(state, action, next_state):
    return 0