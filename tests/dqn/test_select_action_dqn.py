import unittest
import torch
import numpy as np
from src.dqn.select_action_dqn import select_action_dqn
from src.dqn.dqn import DQN

"""
select_action_dqn
input: state, policy_net, epsilon
    policy_net
    input: state (tensor)
    output: distribution (tensor list float)
take the action with maximum value
output: action (tensor number)
"""
states = [torch.Tensor([0,0,0,0]),
          torch.Tensor([0,1,0,0]),
          torch.Tensor([1,0,0,0]),
          torch.Tensor([1,1,0,0])]
distributions = [torch.Tensor([1,0,0,0]),
                 torch.Tensor([0,1,0,0]),
                 torch.Tensor([0,0,1,0]),
                 torch.Tensor([0,0,0,1])]

def dummy_policy_net(state : torch.Tensor) -> torch.Tensor:
    for i in range(0,4):
        if torch.all(torch.eq(states[i], state)):
            return distributions[i]
    return None

class TestSelectAction(unittest.TestCase):
    def setUp(self):
        self.states = [((0,0),(0,0)),
                       ((0,1),(0,0)),
                       ((1,0),(0,0)),
                       ((1,1),(0,0))]
        self.actions = [torch.tensor(0),
                        torch.tensor(1),
                        torch.tensor(2),
                        torch.tensor(3)]
        self.dummy_policy_net = dummy_policy_net
        self.policy_net = DQN()

    def test_normal_states(self):
        for i in range(0,4):
            action_tensor = select_action_dqn(self.states[i], self.dummy_policy_net, 0)
            self.assertEqual(action_tensor.size(), torch.Size([]))
            self.assertTrue(torch.eq(action_tensor, self.actions[i]))
        for i in range(0,4):
            distribution = self.policy_net(states[i])
            ans = np.argmax(distribution.detach())
            action_tensor = select_action_dqn(self.states[i], self.policy_net, 0)
            self.assertTrue(torch.eq(action_tensor, ans))
    
    def test_invalid_states(self):
        invalid_state0 = ((0,0),(0,1))
        self.assertRaises(ValueError, select_action_dqn, invalid_state0, self.dummy_policy_net, 0)
        
    def test_epsilon(self):
        state = self.states[0]
        for policy_net in self.dummy_policy_net, self.policy_net:
            counter = {0:0,1:0,2:0,3:0}
            for i in range(0,100):
                action_tensor = select_action_dqn(state, policy_net, 1)
                action = int(action_tensor)
                if action in counter:
                    counter[action] += 1
                else:
                    print("Unexpected action selection.")
                    self.assertTrue(False)
            for i in range(0,4):
                self.assertGreater(counter[i], 5)
                self.assertLess(counter[i], 50)
            print(counter)

if __name__ == '__main__':
    unittest.main()