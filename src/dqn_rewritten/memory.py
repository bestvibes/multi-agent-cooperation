import random

class PushMemory(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.pointer = 0

    def __call__(self, replay_memory: list, *args) -> list:
        if len(replay_memory) < self.capacity:
            self.pointer = len(replay_memory)
            replay_memory.append(args)
        else:
            self.pointer = (self.pointer+1) % self.capacity
            replay_memory[self.pointer] = (args)
        return replay_memory

def sample_from_memory(memory, size):
    return random.sample(memory, min(size, len(memory)))
