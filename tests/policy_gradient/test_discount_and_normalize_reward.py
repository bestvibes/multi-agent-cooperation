import unittest
import torch
import numpy as np
from src.policy_gradient.discount_and_normalize_reward import discount_and_normalize_reward


class TestComputeLoss(unittest.TestCase):
    def setUp(self):
        self.gamma = 0.95
    
    def test_discount_and_normalize(self):
        rewards = [1.0, 2.0, 3.0]
        last_computed_reward = 2.95 * 0.95 + 3.0
        computed_rewards = [-1.141327628230599, -0.1525432335398914, 1.2938708617704897]
        actual_rewards = list(discount_and_normalize_reward(rewards, self.gamma))
        mean = np.mean(actual_rewards)
        std = np.std(actual_rewards)
        actual_rewards = list((actual_rewards - mean)/std)
        self.assertListEqual(computed_rewards, actual_rewards)


if __name__ == '__main__':
    unittest.main()
        

