from src.dqn_rewritten.DeepQLearning import DeepQLearning
from src.dqn_rewritten.plot import PlotTrainingHistoryCartPole

model_save_path = "dqn_rewritten.st"
fig_save_path = "fig/cartpole_training_history.png"
max_attempt_num = 1

def main():
    transition_function = None
    reward_function = None
    get_initial_state = None
    done_function = None
    render = None
    
    deep_q_learning = DeepQLearning(transition_function,
                                    reward_function,
                                    get_initial_state,
                                    done_function,
                                    render)
    
    net_constructor = None
    max_training_steps = 2000
    max_episode_steps = float('inf')
    learning_rate = 0.005
    batch_size = 30
    eps_max = 1
    eps_min = 0.01
    eps_decay = 0.99
    gamma = 0.99
    memory_capacity = 500
    target_net_update_period = 3

    report_interval = 100
    render_on = False
    target_avg_window = 100
    target_avg = 195
    
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
                                            target_avg)

if __name__ == '__main__':
    attempt_history = []
    for i in range(max_attempt_num):
        print("---- Attempt {} ----".format(i))
        _, training_steps_taken = main()
        attempt_history.append(training_steps_taken)
    print(attempt_history)
    
    
    

