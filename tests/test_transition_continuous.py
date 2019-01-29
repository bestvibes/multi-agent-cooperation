import unittest
from src.transition_continuous import transition_continuous
"""
transition_continous
input: state_space, state, action
    state_space: (x bound, y bound)
    state: ((chaser coord, chaser v), (chasee coord, chasee v))
    action: (delta v on x axis, delta v on y axis)
output: next_state
    next_state: ((new chaser coord, new chaser v), (new chasee coord, new chasee v))
Return the next_state to which action will lead from current state.
In case of out of bound action, stays on the bounded (no reflection)
"""

# dummy definition
#def transition_continuous(state_space, state, action):
#    return (((10,11),(0,1)), ((51,50),(1,0)))

class TestTransitionContinuous(unittest.TestCase):
    def setUp(self):
        self.transition = transition_continuous
        self.state_space = (100, 100)
        self.actions = [(0,0),(0,1),(1,0),(0,-1),(-1,0)]
    
    def test_normal_actions(self):
        curr_state = (((10,10),(0,1)), ((50,50),(1,0)))
        next_states = [(((10,11),(0,1)), ((51,50),(1,0))), # (0,0) action
                       (((10,12),(0,2)), ((51,50),(1,0))), # (0,1)
                       (((11,11),(1,1)), ((51,50),(1,0))), # (1,0)
                       (((10,10),(0,0)), ((51,50),(1,0))), # (0,-1)
                       (((9,11),(-1,1)), ((51,50),(1,0)))  # (-1,0)
                       ]
        for i in range(0,5):
            next_state = self.transition(self.state_space, curr_state, self.actions[i])
            self.assertEqual(next_state, next_states[i])    
        
    def test_out_of_bounds(self):
        curr_state = (((0,0),(0,0.5)), ((99.5,0.5),(1,0.1)))
        next_states = [(((0,0.5),(0,0.5)), ((100,0.6),(1,0.1))), # (0,0) action
                       (((0,1.5),(0,1.5)), ((100,0.6),(1,0.1))), # (0,1)
                       (((1,0.5),(1,0.5)), ((100,0.6),(1,0.1))), # (1,0)
                       (((0,0),(0,-0.5)), ((100,0.6),(1,0.1))),  # (0,-1)
                       (((0,0.5),(-1,0.5)), ((100,0.6),(1,0.1))) # (-1,0)
                       ]
        for i in range(0,5):
            next_state = self.transition(self.state_space, curr_state, self.actions[i])
            self.assertEqual(next_state, next_states[i])
        # extra test cases for boundary
        curr_state2 = (((2,2),(0,0)),((0,0),(0,0)))
        actions2 = [(-4,-3),(-3,-4),(-3,-3),(-3,0),(0,-3)]
        next_states2 = [(((0,0),(-4,-3)),((0,0),(0,0))), # (-4,-3) action
                        (((0,0),(-3,-4)),((0,0),(0,0))), # (-3,-4)
                        (((0,0),(-3,-3)),((0,0),(0,0))),   # (-3,-3)
                        (((0,2),(-3,0)),((0,0),(0,0))),    # (-3,0)
                        (((2,0),(0,-3)),((0,0),(0,0)))     # (0,-3)
                        ]
        for i in range(0,5):
            next_state = self.transition(self.state_space, curr_state2, actions2[i])
            self.assertEqual(next_state, next_states2[i])

if __name__ == '__main__':
    unittest.main()