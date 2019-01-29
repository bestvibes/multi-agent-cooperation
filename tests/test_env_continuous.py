import unittest
from math import sqrt
from src.env_continuous import Env_continuous
"""
Env_continuous class
Initialize:
    state_space: (x bound, y bound)
    action_space: max magnitude of delta v ???
    reward: reward function
    transition: transition function
    done: a function that returns true if terminal state occurs
    start_state: ((chaser coord, chaser v), (chasee coord, chasee v))
    obstacles: unfinished feature
Call:
    input:
        action: delta v = (delta v_x, delta v_y)
    output:
        [action is capped by action_space]
        next_state: as given by transition function
        reward: as given by reward function
        done_condition: as given by done function
"""

from src.reward_continuous import GOAL_REWARD, reward_continuous
from src.transition_continuous import transition_continuous

def done_chasing(state: tuple) -> bool:
    chaser_state = state[0][0]
    chasee_state = state[1][0]
    return chaser_state == chasee_state

# dummy definition
"""
class Env_continuous(object):
    def __init__(self, state_space: tuple,
                       action_space: float,
                       reward: callable,
                       transition: callable,
                       done: callable,
                       start_state: tuple,
                       obstacles: list):
        self.current_state = (((0,0),(0,0)),((0,0),(0,0)))
    def __call__(self, action: tuple) -> (tuple, float, bool):
        next_state = (((0,0),(0,0)),((0,0),(0,0)))
        reward = 0
        done_cond = 0
        return next_state, reward, done_cond
"""
class TestEnvInit(unittest.TestCase):
    def test_bad_start_state(self):
        state_space = (10,10)
        start_state = (((0,1),(100,0)),((20,0),(0,0)))
        action_space = 1
        obstacles = None
        self.assertRaises(ValueError,
                          Env_continuous,
                          state_space,
                          action_space,
                          reward_continuous,
                          transition_continuous,
                          done_chasing,
                          start_state,
                          obstacles)

    def test_normal_start_state(self):
        state_space = (10,10)
        start_state = (((0,1),(100,0)),((10,10),(0,0)))
        action_space = 1
        obstacles = None
        env = Env_continuous(state_space,
                             action_space,
                             reward_continuous,
                             transition_continuous,
                             done_chasing,
                             start_state,
                             obstacles)
        self.assertEqual(start_state, env.current_state)

    def test_bad_action_space(self):
        state_space = (10,10)
        start_state = (((0,1),(100,0)),((10,10),(0,0)))
        action_space = -1
        obstacles = None
        self.assertRaises(ValueError,
                          Env_continuous,
                          state_space,
                          action_space,
                          reward_continuous,
                          transition_continuous,
                          done_chasing,
                          start_state,
                          obstacles)

    def test_wrong_state_dimensions(self):
        state_space = (10,10)
        start_state = (((0,1,1),(100,0)),((10,10),(0,0,0)))
        action_space = 1
        obstacles = None
        self.assertRaises(ValueError,
                          Env_continuous,
                          state_space,
                          action_space,
                          reward_continuous,
                          transition_continuous,
                          done_chasing,
                          start_state,
                          obstacles)

class TestEnvStep(unittest.TestCase):
    def setUp(self):
        state_space = (10, 10)
        action_space = 100
        start_state = (((0,0),(0,0)),((10,10),(0,0)))
        obstacles = None
        self.env = Env_continuous(state_space,
                                  action_space,
                                  reward_continuous,
                                  transition_continuous,
                                  done_chasing,
                                  start_state,
                                  obstacles)

    def test_step(self):
        next_state, reward, done = self.env((5,5))
        self.assertEqual(next_state, (((5,5),(5,5)),((10,10),(0,0))))
        self.assertEqual(reward, -sqrt(50));
        self.assertFalse(done)
        
        next_state, reward, done = self.env((-10,0))
        self.assertEqual(next_state, (((0,10),(-5,5)),((10,10),(0,0))))
        self.assertEqual(reward, -10);
        self.assertFalse(done)
        
        next_state, reward, done = self.env((10,0))
        self.assertEqual(next_state, (((5,10),(5,5)),((10,10),(0,0))))
        self.assertEqual(reward, -5);
        self.assertFalse(done)

        next_state, reward, done = self.env((0,-5))
        self.assertEqual(next_state, (((10,10),(5,0)),((10,10),(0,0))))
        self.assertEqual(reward, GOAL_REWARD);
        self.assertTrue(done)

if __name__ == '__main__':
    unittest.main()