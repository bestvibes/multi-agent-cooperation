import numpy as np
import torch

import src.dqn_rewritten.net as net
from src.dqn_rewritten.policy import Policy
from src.dqn_rewritten.stable_main import model_save_path, ENV_NAME
from src.dqn_rewritten.env import cartpole_render

import gym 
env = gym.make(ENV_NAME)

max_trial_number = 10
max_episode_steps = 500
render_on = True
example_model_path = "example_dqn_rewritten.st"

def evaluate_cartpole_control(model_path):
    trained_q_net = net.Net_4_24_24_2_relu()
    trained_q_net.load_state_dict(torch.load(model_save_path))
    scores = []
    trial_number = 0
    while (trial_number < max_trial_number):
        state = list(env.reset())
        episode_step = 0
        while (episode_step < max_episode_steps):
            if render_on: cartpole_render(state) #env.render()
            policy = Policy(trained_q_net, 0)
            action = policy(state)
            next_state, reward, done, _ = env.step(action.item())
            if done: break
            state = list(next_state)
            episode_step += 1
        scores.append(episode_step)
        trial_number += 1
    print("Evaluation Results:\n\tMean: {}, SD: {:0.3f}\n\tMin: {}, Med: {}, Max: {}"
          .format(np.mean(scores), np.std(scores), np.min(scores), np.median(scores), np.max(scores)))
    return scores

if __name__ == "__main__":
    evaluate_cartpole_control(model_save_path)
    #evaluate_cartpole_control(example_model_path)