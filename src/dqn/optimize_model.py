import torch
import torch.nn.functional as F
from torch.autograd import Variable

import src.util_types as UT
import src.util as U

class ComputeLoss():
    def __init__(self, batch_size, gamma):
        self.batch_size = batch_size
        self.gamma = gamma
    
    def __call__(self, memory, policy_net, target_net):
        
        if len(memory) >= self.batch_size:
            transitions = U.list_batch_random_sample(memory, self.batch_size)
        else:
            transitions  = U.list_batch_random_sample(memory, len(memory))  
            
        batch = UT.Transition(*zip(*transitions))

        state_batch = torch.Tensor([U.flatten_tuple(state_tuple) for state_tuple in batch.state])
        next_state_batch = torch.Tensor([U.flatten_tuple(next_state_tuple) for next_state_tuple in batch.next_state])
        action_batch = torch.LongTensor([action for action in batch.action])
        reward_batch = torch.Tensor([reward for reward in batch.reward])

        print(policy_net(state_batch), action_batch)
        
        state_action_values = policy_net(state_batch).gather(0, action_batch)
        print(policy_net(state_batch), action_batch)

        expected_state_action_values = target_net(next_state_batch).max(0)[0]      
        expected_state_action_values = (torch.unsqueeze(expected_state_action_values, 0) * self.gamma) + reward_batch 
        expected_state_action_values = Variable(expected_state_action_values.data)
        
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        return loss
