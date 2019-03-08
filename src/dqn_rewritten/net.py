# modules for neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_4_24_24_2_relu(nn.Module):
    def __init__(self):
        super(Net_4_24_24_2_relu, self).__init__()
        self.linear1 = nn.Linear(4, 24)
        self.linear2 = nn.Linear(24, 24)
        self.linear3 = nn.Linear(24, 2)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ComputeLoss():
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, sample_memory, q_net, target_net):
        states, actions, next_states, rewards = zip(*sample_memory)
        state_batch = torch.Tensor(states)
        next_state_batch = torch.Tensor(next_states)
        reward_batch = torch.Tensor(rewards)
        action_batch = torch.LongTensor([[action] for action in actions])
        
        state_action_values = q_net(state_batch).gather(1, action_batch)
        
        expected_state_action_values = target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = torch.add((expected_state_action_values * self.gamma), reward_batch)
        expected_state_action_values = torch.Tensor([[value] for value in expected_state_action_values])

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        return loss

def compute_gradient(loss, net):
    net.zero_grad()
    loss.backward()
    gradient = [p.grad.data for p in net.parameters()]
    #print(gradient)
    return gradient

class UpdateNetParameters():
    def __init__(self, learning_rate):
        self.lr = learning_rate
    def __call__(self, net, gradient):
        for p, grad in zip(net.parameters(), gradient):
            p.data.sub_(grad * self.lr)
        return net