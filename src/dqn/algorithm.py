import copy
import collections

import torch

from src.dqn.select_action import select_action_dqn
from src.dqn.compute_loss import ComputeLoss
from src.util import flatten_tuple
from src.algorithm import Algorithm

class DQNPolicy(object):
    def __init__(self,
                    policy_model: torch.nn.Module,
                    parameters,
                    epsilon: float=0.1):
        self.policy_model = policy_model
        self.parameters = parameters
        self.epsilon = epsilon

    def __call__(self, state): # -> action
        return select_action_dqn(state, self.policy_model, self.parameters, self.epsilon)

class TrainDQN(Algorithm):
    def __init__(self,
                    policy_model: torch.nn.Module,
                    initial_parameters: list,
                    batch_size: int=5,
                    gamma: float=0.999,
                    lr=0.001,
                    memory_capacity: int=1000,
                    epsilon: float=1.0,
                    epsilon_decay: float=0.995,
                    epsilon_min: float=0.01,
                    target_network_update_interval_steps: int=10,
                    plot_filename: str="",
                    plot_interval: float=0.01):
        self.target_network_update_interval_steps = target_network_update_interval_steps
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episode_step = 0
        self.training_step = 0

        self.memory = collections.deque(maxlen=memory_capacity)

        self.losses = []
        self.current_training_step_return = 0
        self.returns = []
        self.current_training_step_loss = 0

        self.policy_model = copy.deepcopy(policy_model)
        self.target_parameters = copy.deepcopy(initial_parameters)
        self.parameters = copy.deepcopy(initial_parameters)

        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        self.loss_function = ComputeLoss(batch_size, gamma)

        self.plot_filename = plot_filename
        self.should_plot = plot_filename != ""
        if (self.should_plot):
            from src.plot import PlotLossAndReward
            self.plot = PlotLossAndReward(pause_time=plot_interval, filename=plot_filename)
        else:
            self.plot = None

    def select_action(self, state):
        return DQNPolicy(self.policy_model, self.parameters, self.epsilon)(state)

    def train_episode_step(self, state, action, next_state, reward):
        # store the transition in memory
        state_list = flatten_tuple(state)
        next_state_list = flatten_tuple(next_state)
        self.memory.append((state_list, action, next_state_list, reward))

        # optimize model
        loss = self.loss_function(self.memory, self.policy_model, self.parameters, self.target_parameters)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (self.should_plot):
            self.current_training_step_return += reward
            self.current_training_step_loss += loss.data

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.episode_step += 1

    def train_training_step(self):
        if (self.training_step % self.target_network_update_interval_steps == 0):
            #print(list(map(lambda x,y: x-y, self.parameters, self.target_parameters)))
            self.target_parameters = copy.deepcopy(self.parameters)

        if (self.should_plot):
            self.returns.append(self.current_training_step_return)
            self.current_training_step_return = 0
            self.losses.append(self.current_training_step_loss)
            self.current_training_step_loss = 0
            self.plot(self.losses, self.returns)

        self.training_step += 1

    def get_policy(self):
        return self.parameters
