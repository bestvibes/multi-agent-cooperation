import torch
import numpy as np
from numpy.random import binomial
import random
from src.util import flatten_tuple
    
def select_action_dqn(state : tuple, policy_net, epsilon : float) -> torch.Tensor:
    state_list = flatten_tuple(state)
    state_tensor = torch.Tensor(state_list)
    distribution = policy_net(state_tensor)
    if not isinstance(distribution, torch.Tensor):
        raise ValueError
    random_choice = binomial(1, epsilon)
    if random_choice:
        size = distribution.size()[0]
        action_tensor = torch.tensor(random.randrange(size))
    else:
        action_tensor = np.argmax(distribution.detach())
    return action_tensor