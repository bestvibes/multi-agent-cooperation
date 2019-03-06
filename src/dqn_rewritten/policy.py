import torch
import numpy as np
from numpy.random import binomial
import random

def decay_epsilon(epsilon, eps_decay, eps_min):
    epsilon = max(eps_decay*epsilon, eps_min)
    return epsilon

class Policy():
    def __init__(self, action_value_net, epsilon):
        self.action_value_net = action_value_net
        self.epsilon = epsilon
        
    def __call__(self, state_list):
       state_tensor = torch.tensor(state_list)
       distribution = self.action_value_net(state_tensor)
       if not isinstance(distribution, torch.Tensor):
           raise ValueError
       random_choice = binomial(1, self.epsilon)
       if random_choice:
           size = distribution.size()[0]
           action_tensor = torch.tensor(random.randrange(size))
       else:
           action_tensor = np.argmax(distribution.detach())
       return action_tensor 

def select_action(state_list, q_net, epsilon):
    state_tensor = torch.tensor(state_list)
    distribution = q_net(state_tensor)
    if not isinstance(distribution, torch.Tensor):
        raise ValueError
    random_choice = binomial(1, epsilon)
    if random_choice:
        size = distribution.size()[0]
        action_tensor = torch.tensor(random.randrange(size))
    else:
        action_tensor = np.argmax(distribution.detach())
    return action_tensor