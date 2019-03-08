import unittest

import torch

from src.util_types import Transition
from src.control_with_dqn.dqn import DQN
from src.update_dqn import update_dqn, compute_loss

class TestUpdateDQN(unittest.TestCase):
    def setUp(self):
        self.gamma = 0.99
        self.batch_size = 1
        self.learning_rate = 0.95
    
    def test_loss_decrease(self):
        state = [1,2,3,4]
        next_state = [5,6,7,8]
        reward = -1.0
        action = 1
        # memory = [[state, action, next_state, reward], [state, action, next_state, reward], [state, action, next_state, reward]]
        memory = [Transition(state, action, next_state, reward),Transition(state, action, next_state, reward),Transition(state, action, next_state, reward),Transition(state, action, next_state, reward)]
        
        Q_net = DQN()
        target_net = DQN()

        new_Q_net= update_dqn(Q_net, target_net, memory, self.learning_rate, self.batch_size, self.gamma)
        # loss1 = self.cl(memory, self.policy_net, self.target_net)
        # loss2 = self.cl(memory, self.policy_net, self.target_net)
        # loss1 = compute_loss(Q_net, target_net, memory, self.gamma, self.batch_size)
        # loss2 = compute_loss(new_Q_net, new_target_net, memory, self.gamma, self.batch_size)
        
        # self.assertEqual(loss1, loss2)
        # print(Q_net.state_dict())
        # print(new_Q_net.state_dict())
        test = torch.all(torch.eq(Q_net.state_dict()["linear4.weight"], new_Q_net.state_dict()["linear4.weight"]))

        self.assertTrue(test)

if __name__ == '__main__':
    unittest.main()
        

