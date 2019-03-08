import copy
import time
import random

import numpy as np
import torch

from src.util_types import ACTION_SPACE

from src.plot import PlotLossAndReward
from src.policy_gradient.policy_net import PolicyNet
from src.policy_gradient.loss_function import policy_gradient_loss_function
from src.policy_gradient.select_action import select_action_policy_network
from src.policy_gradient.discount_and_normalize_reward import discount_and_normalize_reward

def policy_gradient_trainer(initial_env,
                start_state: tuple,
                save_path: str,
                gamma: float=0.99,
                learning_rate: float=1e-3,
                epsilon: float=0.1,
                max_training_steps: int=1000,
                max_episode_steps: int=25,
                plot_interval: int=0.01,
                batch_size: int=1,
                random_start_state: bool=False):

    # Initialization
    policy_net = PolicyNet()
    policy_net.load_state_dict(torch.load(save_path))
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    plot_training_curve = PlotLossAndReward(pause_time=plot_interval)

    training_step_reward_history = []
    training_step_loss_history = []
    action_dist_batch = []
    action_batch = []
    reward_batch = []
    # batch_counter = 0

    training_step = 0
    reward_history = []
    step_count_history = []
    # training_step_loss = 0
    while(training_step <= max_training_steps):
        # reset env
        env = copy.deepcopy(initial_env)
        if random_start_state:
            start_state = ((random.randint(-5,5), random.randint(-5,5)),(0, 0))
            env.change_start_state(start_state)

        # action_history = []
        # reward_history = []
        reward_history.append(0)
        step_count_history.append(0)

        state = start_state
        episode_step = 0
        while(episode_step < max_episode_steps):
            action_probabilities = select_action_policy_network(policy_net, state)
            action_distribution = torch.distributions.Categorical(action_probabilities)
            action = action_distribution.sample()


            next_state, reward, done = env(action)

            action_dist_batch.append(action_distribution)
            action_batch.append(action)
            reward_batch.append(reward)
            # action_history.append(action)
            reward_history[-1] += reward

            state = next_state
            episode_step += 1
            step_count_history[-1] += 1

            # print(episode_step, state)

            if done: 
                # training_step_reward_history.append(sum(map(lambda x: x[1], list(zip(action_batch, reward_history)))))
                print("Done in episode", training_step, ", step:", episode_step)
                break

        plot_training_curve(training_step_loss_history, reward_history, step_count_history)

        # update based on the batch size:
        if training_step > 0 and training_step % batch_size == 0:

            # discount and normalize rewards
            discounted_rewards = discount_and_normalize_reward(reward_batch, gamma)

            # optimize model
            optimizer.zero_grad()
            for i in range(len(action_dist_batch)):
                loss = policy_gradient_loss_function(action_dist_batch[i], action_batch[i], torch.Tensor([discounted_rewards[i]]))
                loss.backward(retain_graph=True)
                training_step_loss_history.append(loss.item())
            optimizer.step()

        training_step += 1

        if training_step % 50 == 0:
            print("Training step:", training_step, ", return:", np.mean(reward_history), ", loss", training_step_loss_history[-1], ", start state:", start_state)
            torch.save(policy_net.state_dict(), save_path)

    plot_training_curve(training_step_loss_history, training_step_reward_history)

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
        action = torch.argmax(action_probabilities)
        # action_distribution = torch.distributions.Categorical(action_probabilities)
        # action = action_distribution.sample()
        # action = np.random.choice(ACTION_SPACE, p=action_probabilities)

        next_state, reward, done = env(action)
        state = next_state
        print(i)
        renderer(state)
        time.sleep(0.5)
        if done: break
    