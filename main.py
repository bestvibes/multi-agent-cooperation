import numpy as np
import src.select_action
import src.env
import src.update_q
import src.reward
import src.transition

def main():
    # Initialize environment parameters
    env_size = 1
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    action_space = np.arange(4)
    # Initialize state
    start_state = ((-1, -1), (1, 1))
    
    #print(env(3))

    # Initialize training parameters
    alpha = 0.95
    gamma = 0.95
    max_training_steps = 1000
    max_episode_steps = 25
    epsilon = 0.1 # uncertainy in selecting action
    update_q = src.update_q.UpdateQ(alpha, gamma)
    
    # Initialize Q-table randomly.
    Q_table = {((x1, y1), (x2, y2)):{a: np.random.random() for a in action_space} \
                for x1 in state_space_1D for y1 in state_space_1D for x2 in state_space_1D for y2 in state_space_1D}
    
    # Training
    training_step = 0
    while(training_step <= max_training_steps):
        # Initialize environment
        env = src.env.Env(state_space_bounds,
                          action_space,
                          src.reward.two_agent_chasing_reward_nd_grid,
                          src.transition.transition_2d_grid,
                          start_state)
        state = start_state
        episode_step = 0
        while(episode_step <= max_episode_steps):
            # select action
            action = src.select_action.select_action(state, Q_table, epsilon)

            # update environment
            next_state, reward = env(action)
            
            # update Q-table
            Q_table = update_q(Q_table, state, action, next_state, reward)
            
            if reward == src.reward.GOAL_REWARD:
                print(state, action, next_state, reward, training_step, episode_step)
                break
            
            state = next_state

            episode_step += 1

        training_step += 1
        
    print(Q_table)

if __name__ == '__main__':
    main()