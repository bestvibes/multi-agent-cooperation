import unittest
import torch
from select_action_dqn import select_action_dqn
"""
select_action_dqn
input: state, policy_net, epsilon
    policy_net
    input: state (tensor)
    output: 4d (tensor float)
take the action with maximum value
output: action (tensor, number)
"""
states = [torch.Tensor([0,0,0,0]),
          torch.Tensor([0,1,0,0]),
          torch.Tensor([1,0,0,0]),
          torch.Tensor([1,1,0,0])]
distributions = [torch.Tensor([1,0,0,0]),
                 torch.Tensor([0,1,0,0]),
                 torch.Tensor([0,0,1,0]),
                 torch.Tensor([0,0,0,1])]

def policy_net(state : torch.Tensor) -> torch.Tensor:
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
        self.actions = [torch.tensor([0]),
                        torch.tensor([1]),
                        torch.tensor([2]),
                        torch.tensor([3])]
        self.policy_net = policy_net

    def test_normal_states(self):
        for i in range(0,4):
            action_tensor = select_action_dqn(self.states[i], self.policy_net, 0)
            self.assertEqual(action_tensor.size(), torch.Size([]))
            self.assertTrue(torch.eq(action_tensor, self.actions[i]))
    
    def test_invalid_states(self):
        invalid_state0 = ((0,0),(0,1))
        self.assertRaises(ValueError, select_action_dqn, invalid_state0, self.policy_net, 0)
        
    def test_epsilon(self):
        counter = {0:0,1:0,2:0,3:0}
        state = self.states[0]
        for i in range(0,100):
            action_tensor = select_action_dqn(state, self.policy_net, 1)
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