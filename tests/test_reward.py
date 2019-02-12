import unittest
import math

import src.reward
from src.util_types import ActionCardinal

class TestTwoAgentChasingRewardNdGridWithObstacles(unittest.TestCase):
    def setUp(self):
        self.state_space = ((-2,2),(-2,2))
        self.obstacles = ((1,1), (2,2))
        self.reward_func = src.reward.TwoAgentChasingRewardNdGridWithObstacles(self.state_space, self.obstacles)

    def test_reach_goal_state(self):
        curr_state = ((-1,0), (0,0))
        action = ActionCardinal.RIGHT
        next_state = ((0,0), (0,0))
        reward = self.reward_func(curr_state, action, next_state)

        self.assertEqual(reward, src.reward.GOAL_REWARD)

    def test_l2_dist(self):
        curr_state = ((-1,2), (3,4))
        action = ActionCardinal.DOWN
        next_state = ((-1,1), (3,4))
        reward = self.reward_func(curr_state, action, next_state)

        self.assertEqual(reward, -5)

    def test_out_of_bounds(self):
        curr_state = ((-5,4), (3,4))
        action = ActionCardinal.DOWN
        next_state = ((-5,4), (3,4))
        reward = self.reward_func(curr_state, action, next_state)

        self.assertEqual(reward, src.reward.OUT_OF_BOUNDS_REWARD)

    def test_obstacle(self):
        curr_state = ((1,0), (3,4))
        action = ActionCardinal.UP
        next_state = ((1,1), (3,4))
        reward = self.reward_func(curr_state, action, next_state)

        self.assertEqual(reward, src.reward.OUT_OF_BOUNDS_REWARD)

    def test_no_obstacles(self):
        reward_func = src.reward.TwoAgentChasingRewardNdGridWithObstacles(self.state_space, None)
        curr_state = ((1,0), (3,4))
        action = ActionCardinal.UP
        next_state = ((1,1), (3,4))
        reward = reward_func(curr_state, action, next_state)

        self.assertEqual(reward, -math.sqrt(13))

    def test_invalid_obstacle(self):
        bad_obstacles = ("bad type", (2,2))
        self.assertRaises(ValueError,
                            src.reward.TwoAgentChasingRewardNdGridWithObstacles,
                            self.state_space,
                            bad_obstacles)

    def test_get_off_obstacle(self):
        curr_state = ((2,2), (3,4))
        action = ActionCardinal.DOWN
        next_state = ((2,1), (3,4))
        reward = self.reward_func(curr_state, action, next_state)

        self.assertEqual(reward, -math.sqrt(10))

    def test_obstacle_precendence_over_goal(self):
        curr_state = ((2,1), (2,2))
        action = ActionCardinal.UP
        next_state = ((2,2), (2,2))
        reward = self.reward_func(curr_state, action, next_state)

        self.assertEqual(reward, src.reward.OUT_OF_BOUNDS_REWARD)

if __name__ == '__main__':
    unittest.main()
