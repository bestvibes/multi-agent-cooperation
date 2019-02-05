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

class TestTransitionSingleAgentCardinal(unittest.TestCase):
    def setUp(self):
        state_space = ((-5,5), (-5,5))
        self.transition = src.transition.Transition2dGridSingleAgentCardinal(state_space)
    
    def test_normal_actions(self):
        curr_state = ((0,0), (0,0))
        next_states = {ActionCardinal.UP: ((0,1) , (0,0)),
                       ActionCardinal.DOWN: ((0,-1), (0,0)),
                       ActionCardinal.LEFT: ((-1,0), (0,0)),
                       ActionCardinal.RIGHT: ((1,0) , (0,0)),
                       ActionCardinal.STAY: curr_state}
        for action in ActionCardinal:
            next_state = self.transition(curr_state, action)
            self.assertEqual(next_state, next_states[action])
        
    def test_out_of_bounds(self):
        curr_state = ((5,5), (0,0))
        next_states = {ActionCardinal.UP: curr_state,
                       ActionCardinal.DOWN: ((5,4), (0,0)),
                       ActionCardinal.LEFT: ((4,5), (0,0)),
                       ActionCardinal.RIGHT: curr_state,
                       ActionCardinal.STAY: curr_state}
        for action in ActionCardinal:
            next_state = self.transition(curr_state, action)
            self.assertEqual(next_state, next_states[action])

class TestTransitionMultiAgentCardinal(unittest.TestCase):
    def setUp(self):
        state_space = ((-5,5), (-5,5))
        self.transition = src.transition.Transition2dGridMultiAgentCardinal(state_space)
    
    def test_normal_actions(self):
        curr_state = ((0,0), (1,0))
        actor1_up = (0,1)
        actor1_down = (0,-1)
        actor1_left = (-1,0)
        actor1_right = (1, 0)
        actor1_stay = (0,0)

        actor2_up = (1,1)
        actor2_down = (1,-1)
        actor2_left = (0,0)
        actor2_right = (2, 0)
        actor2_stay = (1,0)

        next_states1 = {ActionCardinal.UP: actor1_up,
                        ActionCardinal.DOWN: actor1_down,
                        ActionCardinal.LEFT: actor1_left,
                        ActionCardinal.RIGHT: actor1_right,
                        ActionCardinal.STAY: actor1_stay}

        next_states2 = {ActionCardinal.UP: actor2_up,
                        ActionCardinal.DOWN: actor2_down,
                        ActionCardinal.LEFT: actor2_left,
                        ActionCardinal.RIGHT: actor2_right,
                        ActionCardinal.STAY: actor2_stay}

        for action1 in ActionCardinal:
            for action2 in ActionCardinal:
                next_state = self.transition(curr_state, [action1, action2])
                self.assertEqual(len(next_state), 2)
                self.assertEqual(next_state[0], next_states1[action1])
                self.assertEqual(next_state[1], next_states2[action2])
        
    def test_out_of_bounds(self):
        curr_state = ((5,5), (-5,0))
        actor1_up = (5,5)
        actor1_down = (5,4)
        actor1_left = (4,5)
        actor1_right = (5, 5)
        actor1_stay = (5,5)

        actor2_up = (-5,1)
        actor2_down = (-5,-1)
        actor2_left = (-5,0)
        actor2_right = (-4, 0)
        actor2_stay = (-5,0)

        next_states1 = {ActionCardinal.UP: actor1_up,
                        ActionCardinal.DOWN: actor1_down,
                        ActionCardinal.LEFT: actor1_left,
                        ActionCardinal.RIGHT: actor1_right,
                        ActionCardinal.STAY: actor1_stay}

        next_states2 = {ActionCardinal.UP: actor2_up,
                        ActionCardinal.DOWN: actor2_down,
                        ActionCardinal.LEFT: actor2_left,
                        ActionCardinal.RIGHT: actor2_right,
                        ActionCardinal.STAY: actor2_stay}

        for action1 in ActionCardinal:
            for action2 in ActionCardinal:
                next_state = self.transition(curr_state, [action1, action2])
                self.assertEqual(len(next_state), 2)
                self.assertEqual(next_state[0], next_states1[action1])
                self.assertEqual(next_state[1], next_states2[action2])

if __name__ == '__main__':
    unittest.main()

