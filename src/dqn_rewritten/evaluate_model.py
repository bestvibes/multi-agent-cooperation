import numpy as np
import torch
import matplotlib.pyplot as plt

import src.dqn_rewritten.net as net
from src.dqn_rewritten.policy import Policy
import src.dqn_rewritten.env as env 

max_trial_number = 30
max_episode_steps = 500
render_on = False
model_save_path = "src/dqn_rewritten/dqn_performance_data/dqn_rewritten5.st"
fig_save_path = "src/dqn_rewritten/dqn_performance_data/eval_model5.png"

render = env.cartpole_render
transition_function = env.cartpole_transition_function
reward_function = env.cartpole_reward_function
get_initial_state = env.cartpole_get_initial_state
done_function = env.cartpole_done_function


def evaluate_cartpole_control(model_path):
    trained_q_net = net.Net_4_24_24_2_relu()
    trained_q_net.load_state_dict(torch.load(model_path))
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
    summary_txt = "Evaluated over 30 episodes:\nMean: {:0.2f}, SD: {:0.2f}\nMin: {}, Med: {}, Max: {}".format(np.mean(scores), np.std(scores), np.min(scores), np.median(scores), np.max(scores))
    print(summary_txt)
    
    plt.hist(scores,30)
    plt.title(summary_txt)
    plt.savefig(fig_save_path)

if __name__ == "__main__":
    evaluate_cartpole_control(model_save_path)
    #evaluate_cartpole_control(example_model_path)