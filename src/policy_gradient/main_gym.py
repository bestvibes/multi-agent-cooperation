import copy
import time
import random

import numpy as np
import torch
import gym

from itertools import count

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.util_types import ACTION_SPACE

from src.plot import PlotLossAndReward
# from src.policy_gradient.policy_net import PolicyNet
from src.policy_gradient.loss_function import policy_gradient_loss_function
from src.policy_gradient.select_action import select_action_policy_network
from src.policy_gradient.discount_and_normalize_reward import discount_and_normalize_reward

from torch import sigmoid
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 1)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def main():
    model_path = "pg_cartpole_v1.st"

    # plot_training_curve = PlotLossAndReward(pause_time=0.01, out_dir="cartpole")

    episode_durations = []
    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        # plt.savefig("plot_cartpole.png")
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig("plot_cartpole_v1.png")

    max_training_steps = 5000
    batch_size = 5
    learning_rate = 0.01
    gamma = 0.99
    # Initialization
    policy_net = PolicyNet()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    training_step_reward_history = []
    training_step_loss_history = []
    action_dist_batch = []
    action_batch = []
    reward_batch = []
    
    env = gym.make('CartPole-v1')

    training_step = 0
    reward_history = []
    step_count_history = []
    
    while(training_step <= max_training_steps):
        state = env.reset()
        # print(state)
        # state = ((state[0], state[1]), (state[2], state[3]))
        state = torch.from_numpy(state).float()
        
        env.render(mode='rgb_array')

        reward_history.append(0)
        step_count_history.append(0)

        for t in count():
            action_probabilities = policy_net(state)
            action_distribution = torch.distributions.Bernoulli(action_probabilities)
            action = action_distribution.sample()

            # print(action)
            action_int = action.data.numpy().astype(int)[0]
            # print(action)
            next_state, reward, done, _ = env.step(action_int)
            env.render(mode='rgb_array')

            if done:
                reward = 0

            action_dist_batch.append(action_distribution)
            action_batch.append(action)
            reward_batch.append(reward)
            
            state = next_state
            state = torch.from_numpy(state).float()
            # state = ((state[0], state[1]), (state[2], state[3]))
            
            if done: 
                episode_durations.append(t + 1)
                plot_durations()
                # plot_training_curve(episode_durations, episode_durations, episode_durations)
                # print(training_step, np.mean(episode_durations))
                break

        # update based on the batch size:
        if training_step > 0 and training_step % batch_size == 0:

            # discount and normalize rewards
            discounted_rewards = discount_and_normalize_reward(reward_batch, gamma)

            # optimize model
            optimizer.zero_grad()
            for i in range(len(action_dist_batch)):
                # loss = policy_gradient_loss_function(action_dist_batch[i], action_batch[i], torch.Tensor([discounted_rewards[i]]))
                loss = -action_dist_batch[i].log_prob(action_batch[i]) * torch.Tensor([discounted_rewards[i]])
                loss.backward()
                # training_step_loss_history.append(loss.item())
            optimizer.step()

            action_dist_batch = []
            action_batch = []
            reward_batch = []

        training_step += 1

        if training_step % 50 == 0:
            torch.save(policy_net.state_dict(), model_path)


if __name__ == "__main__":
    main()
