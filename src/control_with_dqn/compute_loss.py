import torch
import torch.nn.functional as F
from torch.autograd import Variable

import src.util_types as UT
import src.util as U

class ComputeLoss():
    def __init__(self, batch_size=128, gamma=0.9):
        self.batch_size = batch_size
        self.gamma = gamma

    def __call__(self, memory, policy_net, target_net):
        if len(memory) >= self.batch_size:
            transitions = U.list_batch_random_sample(memory, self.batch_size)
        else:
            transitions  = U.list_batch_random_sample(memory, len(memory)) 


        batch = UT.Transition(*zip(*transitions))
        # print(batch)

        state_batch = torch.Tensor(batch.state)
        next_state_batch = torch.Tensor(batch.next_state)
        reward_batch = torch.Tensor(batch.reward)
        action_batch = torch.LongTensor([[action] for action in batch.action])

        state_action_values = policy_net(state_batch).gather(1, action_batch)
        # print(policy_net(state_batch), state_action_values)
        # print(state_action_values)

        # print(next_state_batch)
        # print(target_net(next_state_batch))#.max(1)[0])#.detach())
        expected_state_action_values = target_net(next_state_batch).max(1)[0].detach()
        # print(expected_state_action_values, reward_batch)
        expected_state_action_values = torch.add((expected_state_action_values * self.gamma), reward_batch)
        # print("state"expected_state_action_values)
        expected_state_action_values = torch.Tensor([[value] for value in expected_state_action_values])

        
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        # loss = F.mse_loss(state_action_values, expected_state_action_values)
        # print("v", state_action_values[0], "ex", expected_state_action_values[0], "loss", loss[0])
        return loss