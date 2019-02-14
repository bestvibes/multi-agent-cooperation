import unittest
from unittest.mock import MagicMock
import numpy as np

from src.single_agent_trainer import TrainSingleAgentPolicy
from src.dummy_agent import TrainDummyAgent, DummyCardinalStationaryPolicy

from src.util_types import ActionCardinal

class TestTrainSingleAgentPolicy(unittest.TestCase):
    def setUp(self):
        self.start_state = ((0, 1), (0, 0))
        self.dummy_policy = DummyCardinalStationaryPolicy()
        self.algorithm = TrainDummyAgent(self.dummy_policy)
        self.algorithm.select_action = MagicMock(return_value = ActionCardinal.STAY)
        self.algorithm.train_episode_step = MagicMock()
        self.algorithm.train_training_step = MagicMock()
        self.algorithm.get_policy = MagicMock(return_value=self.dummy_policy)
        self.reward = MagicMock(return_value=0)
        self.transition = MagicMock(return_value=self.start_state)
        self.done = MagicMock(return_value=False)
        self.trainer = TrainSingleAgentPolicy(self.transition, self.reward, self.algorithm)

    def test_train(self):
        trained_policy = self.trainer(self.start_state, self.done, max_training_steps=10, max_episode_steps=5)
        self.assertEqual(self.dummy_policy, trained_policy)

        self.algorithm.select_action.assert_called()
        self.algorithm.train_episode_step.assert_called()
        self.algorithm.train_training_step.assert_called()
        self.algorithm.get_policy.assert_called()

        self.transition.assert_called_with(self.start_state, ActionCardinal.STAY)
        self.done.assert_called_with(self.start_state)
        self.reward.assert_called_with(self.start_state, ActionCardinal.STAY, self.start_state)

if __name__ == '__main__':
    unittest.main()
