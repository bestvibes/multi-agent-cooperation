import unittest
import numpy as np
from update_q import UpdateQ

class TestUpdateQ(unittest.TestCase):
    def setUp(self):
        alpha = 0.95
        gamma = 0.95
        env_size = 5

        self.update = UpdateQ(alpha, gamma)
        self.state_space_1D = range(-env_size, env_size + 1)
        self.action_space = np.arange(4)
    
    def test_computation(self):
        state = ((0, 0), (0, 0))
        next_state = ((1, 0), (0, 0))
        action = 1
        reward = 10.0

        Q_table = {((x1, y1), (x2, y2)): {a: 10.0 for a in self.action_space} for x1 in self.state_space_1D for y1 in self.state_space_1D for x2 in self.state_space_1D for y2 in self.state_space_1D}
        Q_table = self.update(Q_table, state, action, next_state, reward)
        new_value = Q_table[state][action]
        self.assertEqual(new_value, 19.025)


def suite():
    suite = unittest.TestSuite()

    suite.addTest(TestUpdateQ('test_computation'))



    return suite

if __name__ == '__main__':
    # main run

    # run tests
    runner = unittest.TextTestRunner()
    runner.run(suite())
