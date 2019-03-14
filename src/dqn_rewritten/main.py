import numpy as np
import csv

from src.dqn_rewritten.deep_q_learning import DeepQLearning
from src.dqn_rewritten.plot import PlotTrainingHistoryCartPole

import src.dqn_rewritten.env as env
import src.dqn_rewritten.net as net

model_save_path = "dqn_rewritten.st"
fig_save_path = "cartpole_training_history.png"
max_attempt_num = 15


def main():
    transition_function = env.cartpole_transition_function
    reward_function = env.cartpole_reward_function
    get_initial_state = env.cartpole_get_initial_state
    done_function = env.cartpole_done_function
    render = env.cartpole_render
    
    deep_q_learning = DeepQLearning(transition_function,
                                    reward_function,
                                    get_initial_state,
                                    done_function,
                                    render)
    
    net_constructor = net.Net_4_24_24_2_relu
    max_training_steps = 2000
    max_episode_steps = 200
    learning_rate = 0.005
    batch_size = 30
    eps_max = 1
    eps_min = 0.01
    eps_decay = 0.99
    gamma = 0.85
    memory_capacity = 100000
    target_net_update_period = 5

    report_interval = 100
    render_on = False
    target_avg_window = 100
    target_avg = 195
    end_training_if_above_target_avg = True
    
    plot_training_history = PlotTrainingHistoryCartPole(fig_save_path)
    
    action_value_function, training_steps = deep_q_learning(net_constructor,
                                                            max_training_steps, max_episode_steps,
                                                            learning_rate, batch_size,
                                                            eps_max, eps_min, eps_decay,
                                                            gamma,
                                                            memory_capacity,
                                                            target_net_update_period,
                                                            report_interval,
                                                            render_on,
                                                            plot_training_history,
                                                            model_save_path,
                                                            target_avg_window,
                                                            target_avg,
                                                            end_training_if_above_target_avg)
    return training_steps


if __name__ == '__main__':
    attempt_history = []
    for i in range(max_attempt_num):
        print("---- Attempt {} ----".format(i))
        training_steps_taken = main()
        attempt_history.append(training_steps_taken)
    print(attempt_history)
    print("Results:\n\tMean: {}, SD: {:0.3f}\n\tMin: {}, Med: {}, Max: {}"
          .format(np.mean(attempt_history), np.std(attempt_history),
                  np.min(attempt_history), np.median(attempt_history), np.max(attempt_history)))
    with open('cartpole_attempt_history.csv', mode='a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(attempt_history)
    
    

