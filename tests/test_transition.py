import unittest
import src.transition
"""
transition_2d_grid(state_space: tuple, state: tuple, action: int)
state_space: tuple of two tuples, bounds for x, y coordinates
    ((min_x, max_x),(min_y, max_y))
state: current state of the environment
action: the action to be taken
output:
    the next state of the environment (after taking action in current state)
    if the agent intends to go out of bounds, the state does not change
"""

from src.util_types import ActionCardinal, ActionChaserChasee

class TestTransition2DGrid(unittest.TestCase):
    def setUp(self):
        self.transition = src.transition.transition_2d_grid
        self.state_space = ((-5,5), (-5,5))
    
    def test_normal_actions(self):
        curr_state = ((0,0), (0,0))
        next_states = {ActionCardinal.UP: ((0,1) , (0,0)),
                       ActionCardinal.DOWN: ((0,-1), (0,0)),
                       ActionCardinal.LEFT: ((-1,0), (0,0)),
                       ActionCardinal.RIGHT: ((1,0) , (0,0)),
                       ActionCardinal.STAY: curr_state}
        for i in ActionCardinal:
            action = ActionChaserChasee(chaser=i, chasee=ActionCardinal.STAY)
            next_state = self.transition(self.state_space, curr_state, action)
            self.assertEqual(next_state, next_states[i])
        
    def test_out_of_bounds(self):
        curr_state = ((5,5), (0,0))
        next_states = {ActionCardinal.UP: curr_state,
                       ActionCardinal.DOWN: ((5,4), (0,0)),
                       ActionCardinal.LEFT: ((4,5), (0,0)),
                       ActionCardinal.RIGHT: curr_state,
                       ActionCardinal.STAY: curr_state}
        for i in ActionCardinal:
            action = ActionChaserChasee(chaser=i, chasee=ActionCardinal.STAY)
            next_state = self.transition(self.state_space, curr_state, action)
            self.assertEqual(next_state, next_states[i])

if __name__ == '__main__':
    unittest.main()

