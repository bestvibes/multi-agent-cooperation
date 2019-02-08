import torch

from src.dqn.dqn import DQN
from src.dqn.select_action import select_action_dqn
from src.dqn.compute_loss import ComputeLoss
from src.replay_memory import ReplayMemoryPusher
from src.util import flatten_tuple
from src.util_types import Transition
from src.plot import PlotLossAndReward
from src.trainer import Trainer

class DQNPolicy(object):
    def __init__(self,
                    policy_net: DQN,
                    epsilon: float=0.1):
        self.policy_net = policy_net
        self.epsilon = epsilon

    def __call__(self, state): # -> action
        return select_action_dqn(state, self.policy_net, self.epsilon)

def save_dqn(policy: DQNPolicy, save_path: str):
    torch.save(policy.policy_net.state_dict(), save_path)

class DQNAlgorithm(Trainer):
    def __init__(self,
                    batch_size: int=5,
                    gamma: float=0.999,
                    memory_capacity: int=1000,
                    epsilon: float=0.1,
                    target_network_update_interval_steps: int=10,
                    plot_filename: str="",
                    plot_interval: float=0.01):
        self.target_network_update_interval_steps = target_network_update_interval_steps
        self.epsilon = epsilon
        self.episode_step = 0
        self.training_step = 0

        self.memory = []
        self.replay_memory_pusher = ReplayMemoryPusher(Transition, memory_capacity)

        self.losses = []
        self.current_training_step_return = 0
        self.returns = []
        self.current_training_step_loss = 0

        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.loss_function = ComputeLoss(batch_size, gamma)

        self.plot_filename = plot_filename
        self.should_plot = plot_filename != ""
        self.plot = PlotLossAndReward(pause_time=plot_interval, filename=plot_filename) if self.should_plot else None

    def select_action(self, state):
        return select_action_dqn(state, self.policy_net, self.epsilon)

    def train_episode_step(self, state, action, next_state, reward):
        # store the transition in memory
        state_list = flatten_tuple(state)
        next_state_list = flatten_tuple(next_state)
        self.memory = self.replay_memory_pusher(self.memory, state_list, action, next_state_list, reward)

        # optimize model
        loss = self.loss_function(self.memory, self.policy_net, self.target_net)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.current_training_step_return += reward
        self.current_training_step_loss += loss.data

        self.episode_step += 1

    def train_training_step(self):
        if (self.training_step % self.target_network_update_interval_steps == 0):
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.returns.append(self.current_training_step_return)
        self.current_training_step_return = 0

        self.losses.append(self.current_training_step_loss)
        self.current_training_step_loss = 0

        if (self.should_plot):
            self.plot(self.losses, self.returns)

        self.training_step += 1

    def get_policy(self):
        return DQNPolicy(self.policy_net, self.epsilon)
