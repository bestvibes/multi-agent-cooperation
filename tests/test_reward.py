import unittest

import src.reward

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

class TestTwoAgentChasingRewardNdGrid(unittest.TestCase):
    def setUp(self):
        self.reward_func = src.reward.two_agent_chasing_reward_nd_grid

    def test_reach_goal_state(self):
        curr_state = ((-1,0), (0,0))
        action = ACTION_RIGHT
        next_state = ((0,0), (0,0))
        reward = self.reward_func(curr_state, action, next_state)

        self.assertEqual(reward, src.reward.GOAL_REWARD)

    def test_l2_dist(self):
        curr_state = ((-1,2), (3,4))
        action = ACTION_DOWN
        next_state = ((-1,1), (3,4))
        reward = self.reward_func(curr_state, action, next_state)

        self.assertEqual(reward, -5)

    def test_out_of_bounds(self):
        curr_state = ((-5,4), (3,4))
        action = ACTION_DOWN
        next_state = ((-5,4), (3,4))
        reward = self.reward_func(curr_state, action, next_state)

        self.assertEqual(reward, -8)

if __name__ == '__main__':
    unittest.main()