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
        
        state_batch = torch.Tensor(U.flatten_tuple(batch.state[0]))
        next_state_batch = torch.Tensor(U.flatten_tuple(batch.next_state[0]))
        reward_batch = torch.Tensor(batch.reward)
        
        state_value = policy_net(state_batch)
        next_state_value = target_net(next_state_batch)     
        expected_state_value = (next_state_value * self.gamma) + reward_batch 
        expected_state_value = Variable(expected_state_value.data)
        
        loss = F.mse_loss(state_value, expected_state_value)
        return loss
