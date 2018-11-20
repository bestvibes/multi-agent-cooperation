import unittest
import math

import src.env

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

def reward_11_by_11_2d_grid(state: tuple, action: int, next_state: tuple) -> float:
    next_chaser_state = next_state[0]
    next_chasee_state = next_state[1]

    x_dist = next_chaser_state[0] - next_chasee_state[0]
    y_dist = next_chaser_state[1] - next_chasee_state[1]
    
    chaser_chasee_dist = math.sqrt(x_dist**2 + y_dist**2)

    if state == next_state: # if we stepped out of bounds
        return -1000
    elif chaser_chasee_dist == 0:
        return 1000
    else:
        return -chaser_chasee_dist

def transition_11_by_11_2d_grid(state: tuple, action: int) -> tuple:
    grid_x_bounds = (-5, 5)
    grid_y_bounds = (-5, 5)

    current_chaser_state = state[0]
    current_chasee_state = state[1]

    if action == ACTION_RIGHT:
        if (current_chaser_state[0] + 1) <= grid_x_bounds[1]:
            new_chaser_state = (current_chaser_state[0]+1, current_chaser_state[1])
        else:
            new_chaser_state = current_chaser_state
    elif action == ACTION_LEFT:
        if (current_chaser_state[0] - 1) >= grid_x_bounds[0]:
            new_chaser_state = (current_chaser_state[0]-1, current_chaser_state[1])
        else:
            new_chaser_state = current_chaser_state
    elif action == ACTION_UP:
        if (current_chaser_state[1] + 1) <= grid_y_bounds[1]:
            new_chaser_state = (current_chaser_state[0], current_chaser_state[1]+1)
        else:
            new_chaser_state = current_chaser_state
    elif action == ACTION_DOWN:
        if (current_chaser_state[1] - 1) >= grid_y_bounds[0]:
            new_chaser_state = (current_chaser_state[0], current_chaser_state[1]-1)
        else:
            new_chaser_state = current_chaser_state

    return (new_chaser_state, current_chasee_state)

def done_chasing(state: tuple) -> bool:
    chaser_state = state[0]
    chasee_state = state[1]

    return chaser_state == chasee_state

class TestEnvInit(unittest.TestCase):
    def test_bad_start_state(self):
        state_space = ((-5, 5), (-5, 5))
        start_state = ((6,-6), (0,0))
        action_space = [ACTION_DOWN, ACTION_UP, ACTION_LEFT, ACTION_RIGHT]
        self.assertRaises(ValueError,
                            src.env.Env,
                            state_space,
                            action_space,
                            reward_11_by_11_2d_grid,
                            transition_11_by_11_2d_grid,
                            done_chasing,
                            start_state
                            )

    def test_bad_action_space(self):
        state_space = ((-5, 5), (-5, 5))
        start_state = ((0,0), (0,0))
        action_space = [["unhashable type"], ACTION_UP, ACTION_LEFT, ACTION_RIGHT]
        self.assertRaises(ValueError,
                            src.env.Env,
                            state_space,
                            action_space,
                            reward_11_by_11_2d_grid,
                            transition_11_by_11_2d_grid,
                            done_chasing,
                            start_state
                            )

    def test_wrong_state_dimensions(self):
        state_space = ((-5, 5), (-5, 5))
        start_state = ((0,0,0), (0,0,0))
        action_space = [ACTION_DOWN, ACTION_UP, ACTION_LEFT, ACTION_RIGHT]
        self.assertRaises(ValueError,
                            src.env.Env,
                            state_space,
                            action_space,
                            reward_11_by_11_2d_grid,
                            transition_11_by_11_2d_grid,
                            done_chasing,
                            start_state
                            )

class TestEnvStep(unittest.TestCase):
    def setUp(self):
        state_space = ((-5, 5), (-5, 5))
        start_state = ((0,0), (0,0))
        action_space = [ACTION_DOWN, ACTION_UP, ACTION_LEFT, ACTION_RIGHT]
        self.env = src.env.Env(state_space,
                                    action_space,
                                    reward_11_by_11_2d_grid,
                                    transition_11_by_11_2d_grid,
                                    done_chasing,
                                    start_state)

    def test_step(self):
        next_state, reward, done = self.env(ACTION_LEFT)
        self.assertEqual(next_state, ((-1,0), (0,0)))
        self.assertEqual(reward, -1);
        self.assertFalse(done)

        next_state, reward, done = self.env(ACTION_DOWN)
        self.assertEqual(next_state, ((-1,-1), (0,0)))
        self.assertEqual(reward, -math.sqrt(2));
        self.assertFalse(done)

        next_state, reward, done = self.env(ACTION_RIGHT)
        self.assertEqual(next_state, ((0,-1), (0,0)))
        self.assertEqual(reward, -1);
        self.assertFalse(done)

        next_state, reward, done = self.env(ACTION_UP)
        self.assertEqual(next_state, ((0,0), (0,0)))
        self.assertEqual(reward, 1000);
        self.assertTrue(done)

    def test_step_out_of_bounds(self):
        self.env(ACTION_LEFT)
        self.env(ACTION_LEFT)
        self.env(ACTION_LEFT)
        self.env(ACTION_LEFT)
        self.env(ACTION_LEFT)
        next_state, reward, done = self.env(ACTION_LEFT) # should be out of bounds

        self.assertEqual(next_state, ((-5,0), (0,0)))
        self.assertEqual(reward, -1000); # severely penalize going out of bounds
        self.assertFalse(done)

if __name__ == '__main__':
    unittest.main()
