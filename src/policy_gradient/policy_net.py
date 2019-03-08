from torch import sigmoid
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.linear1 = nn.Linear(4, 32)
        self.linear2 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.softmax(x, dim=-1)
        x = torch.sigmoid(x)
        return x