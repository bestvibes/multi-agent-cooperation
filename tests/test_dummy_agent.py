import unittest
import random

import src.dummy_agent

from src.util_types import ActionCardinal
from src.transition import Transition2dGridSingleAgentCardinal

class TestDummyCardinalStationaryPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = src.dummy_agent.DummyCardinalStationaryPolicy()
    
    def test_normal_actions(self):
        env_range = range(-10, 10)
        state_space_generator = (((x,y), (a,b)) for a in env_range for b in env_range for x in env_range for y in env_range)
        for state in state_space_generator:
            action = self.policy(state)
            self.assertEqual(action, ActionCardinal.STAY)

class TestDummyCardinalRandomPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = src.dummy_agent.DummyCardinalRandomPolicy()
    
    def test_normal_actions(self):
        env_range = range(-5, 5)
        state_space_generator = (((x,y), (a,b)) for a in env_range for b in env_range for x in env_range for y in env_range)
        counter = {ActionCardinal.UP:0,
                    ActionCardinal.DOWN:0,
                    ActionCardinal.LEFT:0,
                    ActionCardinal.RIGHT:0,
                    ActionCardinal.STAY:0}
        for state in state_space_generator:
            action = self.policy(state)
            counter[action] += 1

        num_actions = sum(counter.values())
        # each action should have been selected at least 10% of the time (20% each ideally)
        self.assertGreater(counter[ActionCardinal.UP]/num_actions, 0.1)
        self.assertGreater(counter[ActionCardinal.UP]/num_actions, 0.1)
        self.assertGreater(counter[ActionCardinal.UP]/num_actions, 0.1)
        self.assertGreater(counter[ActionCardinal.UP]/num_actions, 0.1)
        self.assertGreater(counter[ActionCardinal.UP]/num_actions, 0.1)

class TestTrainDummyAgent(unittest.TestCase):
    def setUp(self):
        env_size = 10
        state_space_1D = range(-env_size, env_size + 1)
        state_space_bounds = ((-env_size, env_size),)*2
        self.reward = src.dummy_agent.zero_reward
        self.transition = Transition2dGridSingleAgentCardinal(state_space_bounds)
        self.start_state = ((0, 1), (0, 0))
        self.dummy_policy = src.dummy_agent.DummyCardinalStationaryPolicy()
        self.algorithm = src.dummy_agent.TrainDummyAgent(self.dummy_policy)

    def test_train(self):
        for train in range(10):
            state = self.start_state
            for episode in range(5):
                action = self.algorithm.select_action(state)
                next_state = self.transition(state, action)
                reward = self.reward(state, action, next_state)
                self.assertEqual(reward, 0)

                self.algorithm.train_episode_step(state, action, next_state, reward)

                state = next_state

            self.algorithm.train_training_step()

        dummy_policy = self.algorithm.get_policy()
        self.assertEqual(dummy_policy, self.dummy_policy)

if __name__ == '__main__':
    unittest.main()

