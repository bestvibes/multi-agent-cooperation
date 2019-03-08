import torch
import torch.nn.functional as F
from torch.autograd import Variable

import src.util_types as UT
import src.util as U

class ComputeLoss():
    def __init__(self, batch_size=128, gamma=0.9):
        self.batch_size = batch_size
        self.gamma = gamma

    def __call__(self, memory, value_net, target_net):
        if len(memory) >= self.batch_size:
            transitions = U.list_batch_random_sample(memory, self.batch_size)
        else:
            transitions  = U.list_batch_random_sample(memory, len(memory)) 


        batch = UT.Transition(*zip(*transitions))

        state_batch = torch.Tensor(batch.state)
        next_state_batch = torch.Tensor(batch.next_state)
        reward_batch = torch.Tensor(batch.reward)
        # action_batch = torch.LongTensor([[action] for action in batch.action])

        # state_action_values = policy_net(state_batch).gather(1, action_batch)

        # expected_state_action_values = target_net(next_state_batch).max(1)[0].detach()
        # expected_state_action_values = torch.add((expected_state_action_values * self.gamma), reward_batch)
        # expected_state_action_values = torch.Tensor([[value] for value in expected_state_action_values])

        
        state_values = value_net(state_batch).detach()
        print("state_values", state_values)
        expected_state_values = target_net(next_state_batch).detach()
        print("expected_state_values", expected_state_values)
        expected_state_values = torch.add((expected_state_values * self.gamma), reward_batch)
        
        
        
        loss = F.mse_loss(state_values, expected_state_values)
        return loss