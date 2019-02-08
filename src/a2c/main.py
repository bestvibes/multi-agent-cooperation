import copy
import time

import numpy as np
import torch

from src.plot import PlotLossAndReward
from src.util import flatten_tuple
from src.util_types import Transition
from src.replay_memory import ReplayMemoryPusher

from src.a2c.dqn.dqn import DQN
from src.a2c.dqn.optimize_model import ComputeLoss

from src.a2c.policy_net import PolicyNet
from src.a2c.loss_function import policy_gradient_loss_function
from src.a2c.select_action import select_action_policy_network

def a2c_trainer(initial_env,
                start_state: tuple,
                save_path_policy_net: str,
                max_training_steps: int=1000,
                max_episode_steps: int=25,
                # policy net parameters
                gamma_policy_net: float=0.999,
                learning_rate_policy_net: float=0.01,
                # Q net parameters
                gamma_Q_net: float=0.999,
                learning_rate_Q_net: float=1e-4,
                batch_size: int=128,
                memory_capacity: int=1000,
                target_network_update_interval_steps: int=10,
                plot_interval: int=0.01):

    # Initialize Memory
    memory = []
    replay_memory_pusher = ReplayMemoryPusher(Transition, memory_capacity)

    # initialize data for plot
    losses_Q_net = []
    losses_policy_net = []
    returns = []
    
    # Initialize Q_net
    Q_net = DQN()
    target_Q_net = DQN()
    target_Q_net.load_state_dict(Q_net.state_dict())
    target_Q_net.eval()
    optimizer_Q_net = torch.optim.Adam(Q_net.parameters(), lr=learning_rate_Q_net)
    loss_function_Q_net = ComputeLoss(batch_size, gamma_Q_net)
    plot_training_curve_Q_net = PlotLossAndReward(pause_time=plot_interval)
    
    # Initialize policy_net
    policy_net = PolicyNet()
    optimizer_policy_net = torch.optim.Adam(policy_net.parameters(), lr=learning_rate_policy_net)
    plot_training_curve_policy_net = PlotLossAndReward(pause_time=plot_interval)

    training_step = 0
    while(training_step <= max_training_steps):
        # reset env
        env = copy.deepcopy(initial_env)
        state = start_state
        returns.append(0)
        episode_step = 0
        while(episode_step <= max_episode_steps):
            # select action according to policy_net
            action_probabilities = select_action_policy_network(policy_net, state)
            action_distribution = torch.distributions.Categorical(action_probabilities)
            action = action_distribution.sample()
            
            # record feedback from env
            next_state, reward, done = env(action)
            returns[-1] += reward
            
            # update Q_net
            memory = replay_memory_pusher(memory, state, action, next_state, reward)
            loss_Q_net = loss_function_Q_net(memory, Q_net, target_Q_net)
            optimizer_Q_net.zero_grad()
            loss_Q_net.backward()
            optimizer_Q_net.step()
            
            # update policy network
            state_tensor = torch.Tensor(flatten_tuple(state))
            next_state_tensor = torch.Tensor(flatten_tuple(next_state))
            v_state = Q_net(state_tensor)
            v_next_state = Q_net(next_state_tensor)
            loss_policy_net = policy_gradient_loss_function(action_distribution, action, v_state, v_next_state, reward, gamma_policy_net)
            optimizer_policy_net.zero_grad()
            loss_policy_net.backward()
            optimizer_policy_net.step()
            
            # check for terminal conditions
            terminal = (episode_step >= max_episode_steps)
            if done or terminal: 
                losses_Q_net.append(loss_Q_net.data)
                losses_policy_net.append(loss_policy_net.data)
                plot_training_curve_Q_net(losses_Q_net, returns)
                #plot_training_curve_policy_net(losses_policy_net, returns)
                break

            # go to next episode step
            state = next_state
            episode_step += 1

        # update target Q network
        if (training_step % target_network_update_interval_steps == 0):
            target_Q_net.load_state_dict(Q_net.state_dict())

        training_step += 1
        if training_step % 50 == 0:
            print(f"train step: {training_step}")

    torch.save(policy_net.state_dict(), save_path_policy_net)
    #torch.save(Q_net.state_dict(), save_path_Q_net)

def a2c_runner(env,
               start_state: tuple,
               renderer: callable,
               load_path: str,
               max_running_steps: int=25):

    policy_net = PolicyNet()
    policy_net.load_state_dict(torch.load(load_path))
    policy_net.eval()

    state = start_state

    renderer(state)
    for i in range(0, max_running_steps):
        action_probabilities = select_action_policy_network(policy_net, state)
        action_distribution = torch.distributions.Categorical(action_probabilities)
        action = action_distribution.sample()

        next_state, reward, done = env(action)
        state = next_state
        print(i)
        renderer(state)
        time.sleep(0.5)
        if done: break