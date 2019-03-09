import numpy as np
import torch

import src.dqn_rewritten.net as net
from src.dqn_rewritten.policy import Policy
import src.dqn_rewritten.env as env
from src.dqn_rewritten.main import model_save_path

max_trial_number = 10
max_episode_steps = 500
render_on = True
example_model_path = "example_dqn_rewritten.st"

render = env.cartpole_render
transition_function = env.cartpole_transition_function
reward_function = env.cartpole_reward_function
get_initial_state = env.cartpole_get_initial_state
done_function = env.cartpole_done_function


def evaluate_cartpole_control(model_path):
    trained_q_net = net.Net_4_24_24_2_relu()
    trained_q_net.load_state_dict(torch.load(model_save_path))
    scores = []
    trial_number = 0
    while (trial_number < max_trial_number):
        state = list(get_initial_state())
        episode_step = 0
        episode_reward_sum = 0
        while (episode_step < max_episode_steps):
            if render_on: render(state)
            policy = Policy(trained_q_net, 0)
            action = policy(state)
            next_state = transition_function(state, action.item())
            reward = reward_function(state, action.item(), next_state)
            done = done_function(next_state)
            if done: break
            state = list(next_state)
            episode_reward_sum += reward
            episode_step += 1
        scores.append(episode_reward_sum)
        trial_number += 1
    print("Evaluation Results:\n\tMean: {}, SD: {:0.3f}\n\tMin: {}, Med: {}, Max: {}"
          .format(np.mean(scores), np.std(scores), np.min(scores), np.median(scores), np.max(scores)))
    
    return scores

if __name__ == "__main__":
    evaluate_cartpole_control(model_save_path)
    #evaluate_cartpole_control(example_model_path)