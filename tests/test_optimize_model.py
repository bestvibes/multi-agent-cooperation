import unittest
import torch
from src.dqn.optimize_model import ComputeLoss
from src.dqn.dqn import DQN
from src.util_types import Transition


class TestComputerLoss(unittest.TestCase):
    def setUp(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        gamma = 0.95
        batch_size = 3

        self.cl = ComputeLoss(batch_size, gamma)
    
    def test_same_loss(self):
        state = ((1,1), (2,2))
        next_state = ((1,1), (2,2))
        reward = 1
        action = torch.Tensor([2])[0]
        memory = [Transition(state, action, next_state, reward),Transition(state, action, next_state, reward),Transition(state, action, next_state, reward),Transition(state, action, next_state, reward)]
        loss1 = self.cl(memory, self.policy_net, self.target_net)
        loss2 = self.cl(memory, self.policy_net, self.target_net)
        self.assertEqual(loss1, loss2)


if __name__ == '__main__':
    unittest.main()
        

