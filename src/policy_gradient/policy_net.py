import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.linear1 = nn.Linear(4, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.sigmoid(x)
        return x