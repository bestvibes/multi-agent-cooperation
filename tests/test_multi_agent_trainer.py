import unittest
from unittest.mock import MagicMock
import numpy as np

from src.multi_agent_trainer import TrainMultiAgentPolicies
from src.dummy_agent import TrainDummyAgent, DummyCardinalStationaryPolicy

from src.util_types import ActionCardinal

class TestTrainMultiAgentPolicies(unittest.TestCase):
    def setUp(self):
        self.start_state = ((0, 1), (0, 0))

        self.num_agents = 2
        self.dummy_policies = list(map(lambda x: DummyCardinalStationaryPolicy(), range(self.num_agents)))
        self.algorithms = list(map(lambda policy: TrainDummyAgent(policy), self.dummy_policies))
        for i, algorithm in enumerate(self.algorithms):
            algorithm.select_action = MagicMock(return_value = ActionCardinal.STAY)
            algorithm.train_episode_step = MagicMock()
            algorithm.train_training_step = MagicMock()
            algorithm.get_policy = MagicMock(return_value=self.dummy_policies[i])

        self.rewards = list(map(lambda reward: MagicMock(return_value=reward), range(self.num_agents)))

        self.transition = MagicMock(return_value=self.start_state)
        self.done = MagicMock(return_value=False)
        self.trainer = TrainMultiAgentPolicies(self.transition, self.rewards, self.algorithms)

    def test_train(self):
        trained_policies = self.trainer(self.start_state, self.done, max_training_steps=10, max_episode_steps=5)

        list(map(lambda p1, p2: self.assertEqual(p1, p2), trained_policies, self.dummy_policies))
        list(map(lambda algorithm: algorithm.select_action.assert_called(), self.algorithms))
        list(map(lambda algorithm: algorithm.train_episode_step.assert_called(), self.algorithms))
        list(map(lambda algorithm: algorithm.train_training_step.assert_called(), self.algorithms))
        list(map(lambda algorithm: algorithm.get_policy.assert_called(), self.algorithms))

        list(map(lambda r: r.assert_called_with(self.start_state, ActionCardinal.STAY, self.start_state), self.rewards))

        actions = list(map(lambda x: ActionCardinal.STAY, range(self.num_agents)))
        self.transition.assert_called_with(self.start_state, actions)
        self.done.assert_called_with(self.start_state)

if __name__ == '__main__':
    unittest.main()
