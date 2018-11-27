import torch
# import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ComputeLoss():
    def __init__(self, batch_size, gamma):
        self.batch_size = batch_size
        self.gamma = gamma
    
    def __call__(self, memory, policy_net, target_net):
        if len(memory) < self.batch_size:
            return 

        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state, dim=0))
        next_state_batch = Variable(torch.cat(batch.next_state, dim=0))
        reward_batch = Variable(torch.cat(batch.reward, dim=0))
        action_batch = Variable(torch.cat(batch.action, dim=0))

        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = target_net(next_state_batch).max(1)[0]
        expected_state_action_values = (torch.unsqueeze(next_state_values, 1) * self.gamma) + reward_batch
        expected_state_action_values = Variable(expected_state_action_values.data)

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        return loss
