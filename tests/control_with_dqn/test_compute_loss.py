import unittest
import torch
from src.control_with_dqn.compute_loss import ComputeLoss
from src.control_with_dqn.dqn import DQN
from src.util_types import Transition


class TestComputerLoss(unittest.TestCase):
    def setUp(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        gamma = 0.95
        batch_size = 3

        self.cl = ComputeLoss(batch_size, gamma)
    
    def test_same_loss(self):
        tensor = False
        if tensor:
            state = torch.Tensor([1, 2, 3, 4])
            next_state = torch.Tensor([5, 6, 7, 8])
            action = torch.Tensor([1.0])
            reward = torch.Tensor([-1.0])
        else:
            state = [1,2,3,4]
            next_state = [5,6,7,8]
            reward = -1.0
            action = 1
        # memory = [[state, action, next_state, reward], [state, action, next_state, reward], [state, action, next_state, reward]]
        memory = [Transition(state, action, next_state, reward),Transition(state, action, next_state, reward),Transition(state, action, next_state, reward),Transition(state, action, next_state, reward)]
        loss1 = self.cl(memory, self.policy_net, self.target_net)
        loss2 = self.cl(memory, self.policy_net, self.target_net)
        self.assertEqual(loss1, loss2)


if __name__ == '__main__':
    unittest.main()
        

