import torch
from src.util import flatten_tuple

def select_action_policy_network(policy_net, state):
    state_list = flatten_tuple(state)
    state_tensor = torch.Tensor(state_list)
    distribution = policy_net(state_tensor)

    return distribution
