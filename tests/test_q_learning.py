import unittest
import numpy as np
import src.q_learning.algorithm
import src.q_learning.update_q
import src.q_learning.util
from src.transition import Transition2dGridSingleAgentCardinal

from src.util_types import ActionCardinal

class TestRandomQTable(unittest.TestCase):
    def setUp(self):
        self.state_space_1D = range(-1, 2)

    def test_random_q_table(self):
        Q_table = src.q_learning.util.init_random_q_table_2d_square_2_agents(ActionCardinal, self.state_space_1D)
        self.assertEqual(len(Q_table), (3*3) * (3*3))
        for q_values in Q_table.values():
            self.assertEqual(len(q_values), len(ActionCardinal))
            for q_value in q_values.values():
                self.assertTrue(q_value < 1.0)
                self.assertTrue(q_value >= 0)

class TestQLearningPolicy(unittest.TestCase):
    def setUp(self):
        self.state0 = ((0,0),(0,0))
        self.state1 = ((0,1),(0,0))
        qtable = {
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
        self.policy = src.q_learning.algorithm.QLearningPolicy(qtable)

    def test_normal_states(self):
        self.assertEqual(self.policy(self.state0), ActionCardinal.STAY)
        self.assertEqual(self.policy(self.state1), ActionCardinal.UP)

    def test_invalid_states(self):
        invalid_state0 = (0,0)
        invalid_state1 = ((0,0))
        invalid_state2 = ((0,0),(0,1))
        self.assertRaises(KeyError, self.policy, invalid_state0)
        self.assertRaises(KeyError, self.policy, invalid_state1)
        self.assertRaises(KeyError, self.policy, invalid_state2)
    
    def test_epsilon(self):
        counter = {ActionCardinal.UP:0,
                    ActionCardinal.DOWN:0,
                    ActionCardinal.LEFT:0,
                    ActionCardinal.RIGHT:0,
                    ActionCardinal.STAY:0}
        state = self.state0
        self.policy.epsilon = 1
        for i in range(0,100):
            action = self.policy(state)
            if action in counter:
                counter[action] += 1
            else:
                self.assertTrue(False)
        for i in ActionCardinal:
            self.assertGreater(counter[i], 5)
            self.assertLess(counter[i], 50)
        print(counter)

class TestUpdateQ(unittest.TestCase):
    def setUp(self):
        alpha = 0.95
        gamma = 0.95
        env_size = 5

        self.update = src.q_learning.update_q.UpdateQ(alpha, gamma)
        self.state_space_1D = range(-env_size, env_size + 1)
        self.action_type = ActionCardinal
    
    def test_computation(self):
        state = ((0, 0), (0, 0))
        next_state = ((1, 0), (0, 0))
        action = ActionCardinal.LEFT
        reward = 10.0

        Q_table = {((x1, y1), (x2, y2)): {a: 10.0 for a in self.action_type} for x1 in self.state_space_1D for y1 in self.state_space_1D for x2 in self.state_space_1D for y2 in self.state_space_1D}
        Q_table = self.update(Q_table, state, action, next_state, reward)
        new_value = Q_table[state][action]
        self.assertEqual(new_value, 19.025)

class TestQLearningAlgorithm(unittest.TestCase):
    def setUp(self):
        env_size = 5
        state_space_1D = range(-env_size, env_size + 1)
        state_space_bounds = ((-env_size, env_size),)*2
        self.reward = lambda s, a, ns: 10000 if (a == ActionCardinal.LEFT) else -10000
        self.transition = Transition2dGridSingleAgentCardinal(state_space_bounds)
        self.start_state = ((0, 5), (0, 0))
        self.q_table = src.q_learning.util.init_random_q_table_2d_square_2_agents(ActionCardinal, state_space_1D)
        self.trainer = src.q_learning.algorithm.QLearningAlgorithm(self.q_table)

    def test_train(self):
        state = self.start_state
        for train in range(2000):
            for episode in range(100):
                action = self.trainer.select_action(state)
                next_state = self.transition(state, action)
                reward = self.reward(state, action, next_state)

                self.trainer.train_episode_step(state, action, next_state, reward)

                state = next_state

        policy = self.trainer.get_policy()

        counter = {ActionCardinal.UP:0,
                    ActionCardinal.DOWN:0,
                    ActionCardinal.LEFT:0,
                    ActionCardinal.RIGHT:0,
                    ActionCardinal.STAY:0}

        state = self.start_state
        for run in range(100):
            action = policy(state)
            next_state = self.transition(state, action)
            counter[action] += 1
            state = next_state

        print(counter[ActionCardinal.LEFT])
        # at least 25% should be left
        self.assertTrue(counter[ActionCardinal.LEFT] > 25)

if __name__ == '__main__':
    unittest.main()
