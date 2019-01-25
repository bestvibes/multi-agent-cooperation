import os
import unittest
import numpy as np

import src.dqn.main

def done_chasing(state: tuple) -> bool:
    chaser_state = state[0]
    chasee_state = state[1]
    return chaser_state == chasee_state

def mock_renderer(state: tuple):
    return

class TestDQNMain(unittest.TestCase):
    def setUp(self):
        self.model_path = "test_dqn_model_output.st"
        # Environment parameters
        env_size = 1
        state_space_1D = range(-env_size, env_size + 1)
        state_space_bounds = ((-env_size, env_size),)*2
        action_space = np.arange(4)
        obstacles = []
        self.start_state = ((-1, -1), (1, 1))

        self.env = src.env.Env(state_space_bounds,
                          action_space,
                          src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
                          src.transition.transition_2d_grid,
                          done_chasing,
                          self.start_state,
                          obstacles)

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_train(self):
        # just run to see if it doesn't crash for now
        src.dqn.main.dqn_trainer(self.env, self.start_state, self.model_path, batch_size=1, max_training_steps=10)
        self.assertTrue(os.path.isfile(self.model_path))

    def test_runner(self):
        self.test_train() # generate model file to work with
        # just run to see if it doesn't crash for now
        src.dqn.main.dqn_runner(self.env, self.start_state, mock_renderer, self.model_path, max_running_steps=10)


if __name__ == '__main__':
    unittest.main()
