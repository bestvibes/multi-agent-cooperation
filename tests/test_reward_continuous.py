import unittest
from src.reward_continuous import GOAL_REWARD, reward_continuous 
"""
reward_continuous
def two_agent_chasing_reward_nd_grid(state: tuple, action: int, next_state: tuple) -> float:
input: state, action, next_state
output: reward
    reward is the negative l2 distance between the agent and the target
"""

#GOAL_REWARD = 1000
#def reward_continuous(state, action, next_state):
#    return 0.0

class TestRewardContinuous(unittest.TestCase):
    def setUp(self):
        self.state = (((0,0),(0,0)),((0,0),(0,0)))
        self.action = (0,0)
        self.reward_func = reward_continuous

    def test_reach_goal_state(self):
        next_state = (((10,30),(-3,3)), ((10,30),(0,2)))
        reward = self.reward_func(self.state, self.action, next_state)
        self.assertEqual(reward, GOAL_REWARD)

    def test_l2_dist(self):
        next_state = (((0,0),(-3,3)), ((30,40),(0,2)))
        reward = self.reward_func(self.state, self.action, next_state)
        self.assertEqual(reward, -50)

if __name__ == '__main__':
    unittest.main()