import numpy as np
import torch
from src.dqn_rewritten.memory import PushMemory, sample_from_memory
import src.dqn_rewritten.net as net
from src.dqn_rewritten.policy import Policy, decay_epsilon

class DeepQLearning():
    def __init__(self,
                 transition_function,
                 reward_function,
                 get_initial_state,
                 done_function,
                 render):
        self.transition_function = transition_function
        self.done_function = done_function
        self.reward_function = reward_function
        self.get_initial_state = get_initial_state
        self.render = render
    
    def __call__(self,
                 net_constructor,
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
                 end_training_if_above_target_avg):
        memory = []
        push_memory = PushMemory(memory_capacity)
        
        q_net = net_constructor()
        target_net = net_constructor()
        target_net.load_state_dict(q_net.state_dict())
        compute_loss = net.ComputeLoss(gamma)
        update_net_parameters = net.UpdateNetParameters(learning_rate)
    
        epsilon = eps_max
    
        loss_history = []
        episode_length_history = []
        episode_length_avgs = []
        
        training_step = 0
        while (training_step < max_training_steps):
            episode_step = 0
            done = False
            state = self.get_initial_state()
            
            while (episode_step < max_episode_steps and not done):
                if render_on: self.render(state)
                
                policy = Policy(q_net, epsilon)
                action = policy(state)
            
                next_state = self.transition_function(state, action.item())
                reward = self.reward_function(state, action.item(), next_state)
                done = self.done_function(next_state)
            
                memory = push_memory(memory, state, action, next_state, reward)
                sample = sample_from_memory(memory, batch_size)
            
                loss = compute_loss(sample, q_net, target_net)
                loss_history.append(loss.item())
                
                gradient = net.compute_gradient(loss, q_net)
                
                q_net = update_net_parameters(q_net, gradient)
                
                state = next_state
                epsilon = decay_epsilon(epsilon, eps_decay, eps_min)
                episode_step += 1
                
            episode_length_history.append(episode_step)
            latest_lengths_avg = np.mean(episode_length_history[max(-target_avg_window, -len(episode_length_history)):])
            episode_length_avgs.append(latest_lengths_avg)
            if training_step % report_interval == 0:
                print("Training step #{}. Latest Avg Episode Length: {}".format(training_step, latest_lengths_avg))
            
            above_threshold = latest_lengths_avg >= target_avg
            end_training =  above_threshold if end_training_if_above_target_avg else not above_threshold
            if end_training:
                print("Task solved at training step #{}".format(training_step))
                break
        
            training_step += 1
            if training_step % target_net_update_period == 0:
                target_net.load_state_dict(q_net.state_dict())
        
        plot_training_history(loss_history, episode_length_history, episode_length_avgs)
        
        torch.save(q_net.state_dict(), model_save_path)
        
        return q_net, training_step
        
        