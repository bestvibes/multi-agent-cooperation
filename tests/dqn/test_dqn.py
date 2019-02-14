import os
import unittest
import numpy as np

from src.dqn.dqn import DQN
from src.dqn.algorithm import DQNPolicy, TrainDQN
from src.transition import Transition2dGridSingleAgentCardinal
from src.util_types import ActionCardinal

class TestDQNAlgorithm(unittest.TestCase):
    def setUp(self):
        env_size = 1
        state_space_1D = range(-env_size, env_size + 1)
        state_space_bounds = ((-env_size, env_size),)*2
        self.reward = lambda s, a, ns: 10000 if (a == ActionCardinal.LEFT) else -10000
        self.transition = Transition2dGridSingleAgentCardinal(state_space_bounds)
        self.start_state = ((0, 1), (-1, -1))
        policy_net = DQN()
        self.algorithm = TrainDQN(policy_net)

    def test_train(self):
        for train in range(1):
            state = self.start_state
            for episode in range(1):
                action = self.algorithm.select_action(state)
                next_state = self.transition(state, action)
                reward = self.reward(state, action, next_state)

                self.algorithm.train_episode_step(state, action, next_state, reward)

                state = next_state

            self.algorithm.train_training_step()

        trained_dqn = self.algorithm.get_policy()
        policy = DQNPolicy(trained_dqn, epsilon=0)

        counter = {ActionCardinal.UP:0,
                    ActionCardinal.DOWN:0,
                    ActionCardinal.LEFT:0,
                    ActionCardinal.RIGHT:0,
                    ActionCardinal.STAY:0}

        state = self.start_state
        for run in range(100):
            action = ActionCardinal(policy(state))
            next_state = self.transition(state, action)
            counter[action] += 1
            state = next_state

        print(counter)
        # at least 50% should be left
        self.assertGreater(counter[ActionCardinal.LEFT], 50)

if __name__ == '__main__':
    unittest.main()
