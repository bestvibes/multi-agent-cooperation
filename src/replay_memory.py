import random
import copy

from src.util_types import Transition

class ReplayMemoryPusher(object):
    def __init__(self, transition_type: type, capacity: int):
        self.capacity = capacity
        self.transition_type = transition_type

    def __call__(self, replay_memory: list, *args) -> list:
        outp = copy.deepcopy(replay_memory)

        if len(replay_memory) < self.capacity:
            return outp + [self.transition_type(*args)]

        random_index = random.randint(0, self.capacity - 1)
        outp[random_index] = self.transition_type(*args)
        return outp
