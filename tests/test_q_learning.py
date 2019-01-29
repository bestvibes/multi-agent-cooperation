import unittest
import numpy as np
import src.q_learning

from src.util_types import ActionCardinal

class TestRandomQTable(unittest.TestCase):
    def setUp(self):
        self.state_space_1D = range(-1, 2)
        self.start_state = ((0,0), (1,1))

    def test_random_q_table(self):
        Q_table = src.q_learning.init_random_q_table(ActionCardinal, self.state_space_1D, self.start_state)
        self.assertEqual(len(Q_table), (3*3) * (3*3))
        for q_values in Q_table.values():
            self.assertEqual(len(q_values), len(ActionCardinal))
            for q_value in q_values.values():
                self.assertTrue(q_value < 1.0)
                self.assertTrue(q_value >= 0)

def done_chasing(state: tuple) -> bool:
    chaser_state = state[0]
    chasee_state = state[1]
    return chaser_state == chasee_state

class TestQLearningTrainer(unittest.TestCase):
    def setUp(self):
        # Environment parameters
        env_size = 1
        state_space_1D = range(-env_size, env_size + 1)
        state_space_bounds = ((-env_size, env_size),)*2
        obstacles = []
        self.start_state = ((-1, -1), (1, 1))

        self.Q_table = src.q_learning.init_random_q_table(ActionCardinal, state_space_1D, self.start_state)

        self.env = src.env.Env(state_space_bounds,
                          ActionCardinal,
                          src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
                          src.transition.transition_2d_grid,
                          done_chasing,
                          self.start_state,
                          obstacles)

    def test_train(self):
        # just run to see if it doesn't crash for now
        learnt_Q_table = src.q_learning.q_learning_trainer(self.Q_table,
                                                            self.env,
                                                            self.start_state,
                                                            max_training_steps=50)
        self.assertTrue(isinstance(learnt_Q_table, dict))

def mock_renderer(state: tuple):
    return

class TestQLearningRunner(unittest.TestCase):
    def setUp(self):
        # Environment parameters
        env_size = 1
        state_space_1D = range(-env_size, env_size + 1)
        state_space_bounds = ((-env_size, env_size),)*2
        obstacles = []
        self.start_state = ((-1, -1), (1, 1))

        self.Q_table = src.q_learning.init_random_q_table(ActionCardinal, state_space_1D, self.start_state)

        self.env = src.env.Env(state_space_bounds,
                          ActionCardinal,
                          src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
                          src.transition.transition_2d_grid,
                          done_chasing,
                          self.start_state,
                          obstacles)

    def test_runner(self):
        # just run to see if it doesn't crash for now
        src.q_learning.q_learning_runner(self.Q_table,
                                            self.env,
                                            self.start_state,
                                            mock_renderer,
                                            max_running_steps=10,
                                            render_interval=0)


if __name__ == '__main__':
    unittest.main()