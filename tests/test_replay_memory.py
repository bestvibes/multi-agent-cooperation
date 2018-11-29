import unittest
from collections import namedtuple

import src.replay_memory

TestType = namedtuple('Test', ('value'))

class TestReplayMemoryPusher(unittest.TestCase):
    def test_push_to_capacity(self):
        replay_memory = []
        capacity = 5
        pusher = src.replay_memory.ReplayMemoryPusher(TestType, capacity)
        for i in range(capacity):
            replay_memory = pusher(replay_memory, i)

        self.assertEqual(list(map(TestType, [0, 1, 2, 3, 4])), replay_memory)

    def test_push_over_capacity(self):
        replay_memory = [10, 20, 30, 40, 50]
        capacity = 5
        pusher = src.replay_memory.ReplayMemoryPusher(TestType, capacity)
        for i in range(capacity):
            replay_memory = pusher(replay_memory, i)
            self.assertTrue(TestType(i) in replay_memory and len(replay_memory) == capacity)
