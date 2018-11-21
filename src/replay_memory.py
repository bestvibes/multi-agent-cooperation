import random
import copy

class ReplayMemoryPusher(object):
    def __init__(self, capacity: int):
        self.capacity = capacity

    def __call__(self, replay_memory: list, *args) -> list:
        outp = copy.deepcopy(replay_memory)

        if len(replay_memory) < self.capacity:
            return outp + [Transition(*args)]

        random_index = random.randint(0, self.capacity - 1)
        outp[random_index] = Transition(*args)
        return outp
