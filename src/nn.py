import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

def create_init_linear_layer_weights_and_bias(n_in, n_out):
    # Logic taken from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L58
    weights = torch.zeros(n_out, n_in, requires_grad=True)
    bias = torch.zeros(n_out, requires_grad=True)

    init.kaiming_uniform_(weights, a=math.sqrt(5))

    fan_in, _ = init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(bias, -bound, bound)

    return Parameter(weights), Parameter(bias)

class ReluSoftmaxNN(nn.Module):
    def __init__(self, layers):
        super(ReluSoftmaxNN, self).__init__()
        # need to set each layer as an attribute to get registered
        # in pytorch
        self.num_layers = len(layers)
        self.num_parameters = 2*self.num_layers # weight+bias per layer
        for i, layer in enumerate(layers):
            self.__setattr__(f"layer{i}", layer)

    def forward(self, x, parameters):
        assert(len(parameters) == self.num_parameters)
        for i in range(self.num_layers-1): # skip last output layer
            x = F.relu(getattr(self, f"layer{i}")(x, parameters[2*i], bias=parameters[2*i+1]))

        return F.softmax(getattr(self, f"layer{self.num_layers-1}")(x, parameters[-2], bias=parameters[-1]))

class ReluLinearNN(nn.Module):
    def __init__(self, layers):
        super(ReluLinearNN, self).__init__()
        # need to set each layer as an attribute to get registered
        # in pytorch
        self.num_layers = len(layers)
        self.num_parameters = 2*self.num_layers # weight+bias per layer
        for i, layer in enumerate(layers):
            self.__setattr__(f"layer{i}", layer)

    def forward(self, x, parameters):
        assert(len(parameters) == self.num_parameters)
        for i in range(self.num_layers-1): # skip last output layer
            x = F.relu(getattr(self, f"layer{i}")(x, parameters[2*i], bias=parameters[2*i+1]))

        return getattr(self, f"layer{self.num_layers-1}")(x, parameters[-2], bias=parameters[-1])
