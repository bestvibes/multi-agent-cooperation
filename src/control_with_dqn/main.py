import copy
import time

import torch

from src.replay_memory import ReplayMemoryPusher
from src.control_with_dqn.dqn import DQN
from src.plot import PlotLossAndReward
from src.control_with_dqn.compute_loss import ComputeLoss
from src.control_with_dqn.select_action import select_action_dqn
from src.util_types import Transition
from src.util import flatten_tuple

def dqn_trainer_control(initial_env,
                start_state: tuple,
                save_path: str,
                learning_rate: float=1e-4,
                batch_size: int=5,
                gamma: float=0.999,
                memory_capacity: int=1000,
                max_training_steps: int=1000,
                max_episode_steps: int=25,
                epsilon: int=0.1,
                target_network_update_interval_steps: int=10,
                plot_interval: float=0.01):

    # Initialization
    memory = []
    replay_memory_pusher = ReplayMemoryPusher(Transition, memory_capacity)

    # For Plotting training curve
    losses = []
    returns = []

    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters())
    loss_function = ComputeLoss(batch_size, gamma)
    plot_training_curve = PlotLossAndReward(pause_time=plot_interval)

    training_step = 0
    while(training_step <= max_training_steps):
        # reset env
        env = copy.deepcopy(initial_env)

        returns.append(0)
        state = start_state
        episode_step = 0
        while(episode_step <= max_episode_steps):
            action = select_action_dqn(state, policy_net, epsilon)

            next_state, reward, done = env(action)
            returns[-1] += reward

            # store the transition in memory
            state_list = flatten_tuple(state)
            next_state_list = flatten_tuple(next_state)
            memory = replay_memory_pusher(memory, state_list, action, next_state_list, reward)

            # optimize model
            loss = loss_function(memory, policy_net, target_net)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            terminal = (episode_step >= max_episode_steps)
            if done or terminal: 
                losses.append(loss.data)
                plot_training_curve(losses, returns)
                break

            state = next_state
            episode_step += 1

        # update target network
        if (training_step % target_network_update_interval_steps == 0):
            target_net.load_state_dict(policy_net.state_dict())

        training_step += 1
        if training_step % 1 == 0:
            print(f"train step: {training_step}")

    torch.save(policy_net.state_dict(), save_path)

def dqn_runner_control(env,
                start_state: tuple,
                renderer: callable,
                load_path: str,
                max_running_steps: int=25,
                epsilon: int=0.1):

    policy_net = DQN()
    policy_net.load_state_dict(torch.load(load_path))
    policy_net.eval()

    state = start_state

    renderer(state)
    for i in range(0, max_running_steps):
        action = select_action_dqn(state, policy_net, epsilon)
        next_state, reward, done = env(action)
        state = next_state
        print(i)
        renderer(state)
        time.sleep(0.5)
        if done: break
    