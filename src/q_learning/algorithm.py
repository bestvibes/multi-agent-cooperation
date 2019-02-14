from src.q_learning.select_action import select_action as q_learning_select_action
from src.q_learning.update_q import UpdateQ
from src.algorithm import Algorithm

class QLearningPolicy(object):
    def __init__(self,
                    q_table: dict,
                    epsilon: float=0.1):
        self.q_table = q_table
        self.epsilon = epsilon

    def __call__(self, state): # -> action
        return q_learning_select_action(state, self.q_table, self.epsilon)

class TrainQTable(Algorithm):
    def __init__(self,
                    initial_q_table: dict,
                    alpha: float=0.95,
                    gamma: float=0.95,
                    epsilon: float=0.1):
        self.q_table = initial_q_table
        self.epsilon = epsilon
        self.update_q = UpdateQ(alpha, gamma)
        self.episode_step = 0
        self.training_step = 0

    def select_action(self, state):
        return QLearningPolicy(self.q_table, self.epsilon)(state)

    def train_episode_step(self, state, action, next_state, reward):
        self.q_table = self.update_q(self.q_table, state, action, next_state, reward)
        self.episode_step += 1

    def train_training_step(self):
        self.training_step += 1

    def get_policy(self):
        return self.q_table
