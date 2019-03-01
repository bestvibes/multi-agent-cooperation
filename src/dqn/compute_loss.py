import torch
import torch.nn.functional as F
from torch.autograd import Variable

import src.util as U

class ComputeLoss():
    def __init__(self, batch_size=128, gamma=0.9):
        self.batch_size = batch_size
        self.gamma = gamma

    def __call__(self, memory, policy_model, parameters, target_parameters):
        if len(memory) >= self.batch_size:
            transitions = U.list_batch_random_sample(memory, self.batch_size)
        else:
            transitions  = U.list_batch_random_sample(memory, len(memory)) 

        states, actions, next_states, rewards = zip(*transitions)

        state_batch = torch.Tensor(states)
        next_state_batch = torch.Tensor(next_states)
        reward_batch = torch.Tensor(rewards)
        action_batch = torch.LongTensor([[action] for action in actions])

        state_action_values = policy_model(state_batch, parameters).gather(1, action_batch)

        expected_state_action_values = policy_model(next_state_batch, target_parameters).max(1)[0].detach()
        expected_state_action_values = torch.add((expected_state_action_values * self.gamma), reward_batch)
        expected_state_action_values = torch.Tensor([[value] for value in expected_state_action_values])

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        return loss