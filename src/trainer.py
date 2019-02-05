from abc import ABC, abstractmethod
 
class Trainer(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def select_action(self, state): # -> action
        pass

    @abstractmethod
    def train_episode_step(self, state, action, next_state, reward):
        pass

    def train_training_step(self):
        pass

    @abstractmethod
    def get_policy(self): # -> policy
        pass
