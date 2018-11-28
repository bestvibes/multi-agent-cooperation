import torch
import numpy as np
from numpy.random import binomial
import random
"""
select_action_dqn
input: state, policy_net, epsilon
    policy_net
    input: state (tensor)
    output: 4d (tensor float)
take the action with maximum value
output: action (tensor, number)
"""
def flatten_tuple(t: tuple):
    return list(sum(t, ()))
    
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
        action_tensor = np.argmax(distribution)
    return action_tensor