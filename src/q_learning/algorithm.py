from src.q_learning.select_action import select_action as q_learning_select_action
from src.q_learning.update_q import UpdateQ
from src.trainer import Trainer

class QLearningPolicy(object):
    def __init__(self,
                    q_table: dict,
                    epsilon: float=0.1):
        self.q_table = q_table
        self.epsilon = epsilon

    def __call__(self, state): # -> action
        return q_learning_select_action(state, self.q_table, self.epsilon)

class QLearningAlgorithm(Trainer):
    def __init__(self,
                    initial_q_table: dict,
                    alpha: float=0.95,
                    gamma: float=0.95,
                    epsilon: float=0.1):
        self.policy = QLearningPolicy(initial_q_table, epsilon)
        self.epsilon = epsilon
        self.update_q = UpdateQ(alpha, gamma)
        self.episode_step = 0
        self.training_step = 0

    def select_action(self, state):
        return self.policy(state)

    def train_episode_step(self, state, action, next_state, reward):
        self.policy.q_table = self.update_q(self.policy.q_table, state, action, next_state, reward)
        self.episode_step += 1

    def train_training_step(self):
        self.training_step += 1

    def get_policy(self):
        return self.policy
