import numpy as np

# modules for handling memory
from src.replay_memory import ReplayMemoryPusher
from src.dqn_rewritten.sample_from_memory import sample_from_memory
from src.util_types import Transition

# modules for neural network
import torch
import src.dqn_rewritten.net as net

# modules for selecting action
from src.dqn_rewritten.select_action import select_action, decay_epsilon

# modules for env
ENV_NAME = "CartPole-v1"
import gym

# modules for visualization
from src.dqn_rewritten.plot import PlotTrainingHistoryCartPole

# modules for evaluating model
from src.dqn_rewritten.evaluate_model import evaluate

# hyperparameters
max_training_steps = 1 #2000
max_episode_steps = 5 #float('inf')
learning_rate = 0.005
batch_size = 30
eps_max = 1
eps_min = 0.01
eps_decay = 0.99
gamma = 0.99
memory_capacity = 500
target_net_update_period = 3

# misc
AVG_WINDOW = 100
AVG_THRESHOLD = 195
model_save_path = "dqn_rewritten.st"
fig_save_path = "fig"
report_interval = 100
max_attempt_num = 1

def main(eval_only=False):
    env = gym.make(ENV_NAME)
    
    if eval_only:
        trained_q_net = net.Net()
        trained_q_net.load_state_dict(torch.load(model_save_path))
        scores = evaluate(trained_q_net, select_action, env, max_trial_number=50, max_episode_steps=500)
        print("Evaluation Results:\n\tMean: {}, SD: {:0.3f}\n\tMin: {}, Med: {}, Max: {}"
              .format(np.mean(scores), np.std(scores), np.min(scores), np.median(scores), np.max(scores)))
        return
    
    memory = []
    replay_memory_pusher = ReplayMemoryPusher(Transition, memory_capacity)
    
    # Change: Net class, seperate architecture and parameters
    q_net = net.Net()
    target_net = net.Net()
    target_net.load_state_dict(q_net.state_dict())
    compute_loss = net.ComputeLoss(gamma)
    update_net_parameters = net.UpdateNetParameters(learning_rate)
    
    epsilon = eps_max
    
    loss_history = []
    score_history = []
    score_avgs = []
    plot_training_history = PlotTrainingHistoryCartPole(fig_save_path)
    
    training_step = 0
    while (training_step < max_training_steps):
        # begining of new episode, reset env
        episode_step = 0
        done = False
        state = list(env.reset())
        while (episode_step < max_episode_steps and not done):
            #env.render()
            
            # Change: polocy(state)->action
            action = select_action(state, q_net, epsilon)
            
            next_state, reward, done, _ = env.step(action.item())
            next_state = list(next_state)
            # Change: not obvious
            reward = reward if not done else -reward
            
            memory = replay_memory_pusher(memory, state, action, next_state, reward)
            sample = sample_from_memory(memory, batch_size)
            
            # compute loss
            loss = compute_loss(sample, q_net, target_net)
            loss_history.append(loss.item())
            
            # compute gradient
            gradient = net.compute_gradient(loss, q_net)
            
            # update parameters
            q_net = update_net_parameters(q_net, gradient)
            
            # test loss monotonicity
            """
            copy_net = net.Net()
            copy_net.load_state_dict(q_net.state_dict())
            for i in range(0,5):
                new_loss = compute_loss(sample, copy_net, target_net)
                print("Loss after update: {}".format(new_loss.item()))
                copy_net = update_net_parameters(copy_net, gradient)
                if loss < new_loss:
                    print("Warning: gradient descent failed to decrease loss")
                loss = new_loss
            """
            
            state = next_state
            epsilon = decay_epsilon(epsilon, eps_decay, eps_min)
            episode_step += 1
        
        score_history.append(episode_step)
        latest_scores_avg = np.mean(score_history[max(-AVG_WINDOW, -len(score_history)):])
        score_avgs.append(latest_scores_avg)
        if training_step % report_interval == 0:
            print("training step #{}. Latest Avg Score: {}".format(training_step, latest_scores_avg))
        if latest_scores_avg >= AVG_THRESHOLD:
            print("Cart-pole task solved at training step #{}".format(training_step))
            break
        
        training_step += 1
        if training_step % target_net_update_period == 0:
            target_net.load_state_dict(q_net.state_dict())
        
    plot_training_history(loss_history, score_history, score_avgs)
    
    torch.save(q_net.state_dict(), model_save_path)
    
    return(training_step)

if __name__ == '__main__':
    attempt_history = []
    for i in range(max_attempt_num):
        print("---- Attempt {} ----".format(i))
        training_steps_taken = main(eval_only=False)
        attempt_history.append(training_steps_taken)
    print(attempt_history)
# lr=0.05, [547, 414, 772, 585, 713, 658, 332, 370, 684, 446]