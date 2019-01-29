import copy
import time

import numpy as np
import torch

from src.plot import PlotLossAndReward
from src.policy_gradient.policy_net import PolicyNet
from src.policy_gradient.loss_function import policy_gradient_loss_function
from src.policy_gradient.select_action import select_action_policy_network
from src.util_types import ActionCardinal, ActionChaserChasee

def policy_gradient_trainer(initial_env,
                start_state: tuple,
                save_path: str,
                gamma: float=0.999,
                learning_rate: float=0.01,
                max_training_steps: int=1000,
                max_episode_steps: int=25,
                plot_interval: int=0.01):

    # Initialization
    policy_net = PolicyNet()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    plot_training_curve = PlotLossAndReward(pause_time=plot_interval)

    training_step_reward_history = []
    training_step_loss_history = []

    training_step = 0
    while(training_step <= max_training_steps):
        # reset env
        env = copy.deepcopy(initial_env)

        step_history = []

        state = start_state
        episode_step = 0
        while(episode_step <= max_episode_steps):
            action_probabilities = select_action_policy_network(policy_net, state)
            action_distribution = torch.distributions.Categorical(action_probabilities)
            action = action_distribution.sample()

            action_chaser = ActionCardinal(action.item())
            action_chaser_chasee = ActionChaserChasee(action_chaser, ActionCardinal.STAY)

            next_state, reward, done = env(action_chaser_chasee)

            step_history.append((action, reward))

            if done: break

            state = next_state
            episode_step += 1

        R = 0
        updated_step_history = []
        for action, r in step_history:
            R = r + R*gamma
            updated_step_history.append((action, R))

        normalized_step_history = []
        r_mean = np.mean(list(map(lambda x: x[1], updated_step_history)))
        r_stdev = np.std(list(map(lambda x: x[1], updated_step_history)))
        for action, r in updated_step_history:
            normalized_step_history.append((action, float(r - r_mean)/r_stdev))

        # optimize model
        training_step_loss = 0
        for action, reward in normalized_step_history:
            training_step_loss += policy_gradient_loss_function(action_distribution, action, reward)
        # print(f"loss: {training_step_loss}")
        training_step_loss_history.append(training_step_loss.data)
        training_step_reward_history.append(sum(map(lambda x: x[1], normalized_step_history)))
        #plot_training_curve(training_step_loss_history, training_step_reward_history)
        optimizer.zero_grad()
        training_step_loss.backward()
        optimizer.step()
        training_step += 1
        print(f"train step: {training_step}")

    torch.save(policy_net.state_dict(), save_path)

def policy_gradient_runner(env,
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

        action_chaser = ActionCardinal(action.item())
        action_chaser_chasee = ActionChaserChasee(action_chaser, ActionCardinal.STAY)

        next_state, reward, done = env(action_chaser_chasee)
        state = next_state
        print(i)
        renderer(state)
        time.sleep(0.5)
        if done: break
    