import os
import unittest
import numpy as np

from src.dqn.dqn import DQN
from src.dqn.algorithm import DQNPolicy, DQNAlgorithm
from src.transition import Transition2dGridSingleAgentCardinal
from src.util_types import ActionCardinal

# class TestDQNMain(unittest.TestCase):
#     def setUp(self):
#         self.model_path = "test_dqn_model_output_control.st"
#         # Environment parameters
#         env_size = 1
#         state_space_1D = range(-env_size, env_size + 1)
#         state_space_bounds = ((-env_size, env_size),)*2
#         action_space = np.arange(4)
#         obstacles = []
#         self.start_state = ((-1, -1), (1, 1))

#         self.env = src.env.Env(state_space_bounds,
#                           action_space,
#                           src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
#                           src.transition.transition_2d_grid,
#                           done_chasing,
#                           self.start_state,
#                           obstacles)

#     def tearDown(self):
#         if os.path.exists(self.model_path):
#             os.remove(self.model_path)

#     def test_train(self):
#         # just run to see if it doesn't crash for now
#         src.control_with_dqn.main.dqn_trainer_control(self.env, self.start_state, self.model_path, batch_size=1, max_training_steps=10)
#         self.assertTrue(os.path.isfile(self.model_path))

#     def test_runner(self):
#         self.test_train() # generate model file to work with
#         # just run to see if it doesn't crash for now
#         src.control_with_dqn.main.dqn_runner_control(self.env, self.start_state, mock_renderer, self.model_path, max_running_steps=10)

class TestDQNAlgorithm(unittest.TestCase):
    def setUp(self):
        env_size = 1
        state_space_1D = range(-env_size, env_size + 1)
        state_space_bounds = ((-env_size, env_size),)*2
        self.reward = lambda s, a, ns: 10000 if (a == ActionCardinal.LEFT) else -10000
        self.transition = Transition2dGridSingleAgentCardinal(state_space_bounds)
        self.start_state = ((0, 1), (-1, -1))
        self.trainer = DQNAlgorithm()

    def test_train(self):
        for train in range(100):
            state = self.start_state
            for episode in range(50):
                action = self.trainer.select_action(state)
                next_state = self.transition(state, action)
                reward = self.reward(state, action, next_state)

                self.trainer.train_episode_step(state, action, next_state, reward)

                state = next_state

            self.trainer.train_training_step()

        policy = self.trainer.get_policy()

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
