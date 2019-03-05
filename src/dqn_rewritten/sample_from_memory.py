import random

def sample_from_memory(memory, size):
    return random.sample(memory, min(size, len(memory)))