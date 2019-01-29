import unittest
from src.select_action import select_action
from src.util_types import ActionCardinal
"""
select_action(state : tuple, qtable : dict, epsilon=0)
state: state of the environment, for now (coordinates of chaser, coordinates of target)
qtable: a dictionary mapping from a state to another dictionary, action2qvalue
    qtable: (state, action2qvalue)
    action2qvalue: (action, qvalue)
output: action in [0, 1, 2, 3]
    with probabilty = 1 - epsilon, action is argmax qtable[state]
    with probabilty = epsilon, action is randomly chosen from {0, 1, 2, 3}
"""
class TestSelectAction(unittest.TestCase):
    def setUp(self):
        self.state0 = ((0,0),(0,0))
        self.state1 = ((0,1),(0,0))
        self.qtable = {
                #   state     : { action : qvalues, ... }
                self.state0 : {ActionCardinal.UP:0.0,
                                ActionCardinal.DOWN:0.1,
                                ActionCardinal.LEFT:0.2,
                                ActionCardinal.RIGHT:0.3,
                                ActionCardinal.STAY:0.4},
                self.state1 : {ActionCardinal.UP:0.4,
                                ActionCardinal.DOWN:0.3,
                                ActionCardinal.LEFT:0.2,
                                ActionCardinal.RIGHT:0.1,
                                ActionCardinal.STAY:0.0}
                }

    def test_normal_states(self):
        self.assertEqual(select_action(self.state0, self.qtable), ActionCardinal.STAY)
        self.assertEqual(select_action(self.state1, self.qtable), ActionCardinal.UP)

    def test_invalid_states(self):
        invalid_state0 = (0,0)
        invalid_state1 = ((0,0))
        invalid_state2 = ((0,0),(0,1))
        self.assertRaises(KeyError, select_action, invalid_state0, self.qtable)
        self.assertRaises(KeyError, select_action, invalid_state1, self.qtable)
        self.assertRaises(KeyError, select_action, invalid_state2, self.qtable)
    
    def test_epsilon(self):
        counter = {ActionCardinal.UP:0,
                    ActionCardinal.DOWN:0,
                    ActionCardinal.LEFT:0,
                    ActionCardinal.RIGHT:0,
                    ActionCardinal.STAY:0}
        state = self.state0
        for i in range(0,100):
            action = select_action(state, self.qtable, 1)
            if action in counter:
                counter[action] += 1
            else:
                self.assertTrue(False)
        for i in ActionCardinal:
            self.assertGreater(counter[i], 5)
            self.assertLess(counter[i], 50)
        print(counter)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSelectAction('test_normal_states'))
    suite.addTest(TestSelectAction('test_invalid_states'))
    suite.addTest(TestSelectAction('test_epsilon'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())