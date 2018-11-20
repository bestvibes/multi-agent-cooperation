import numpy as np
import time
import src.select_action
import src.env
import src.update_q
import src.reward
import src.transition

class Render():
    def __init__(self, chaser_pos, target_pos, env_size):
        self.env_size = env_size
        self.target_y = target_pos[1] + self.env_size
        self.target_x = self.env_size - target_pos[0]
        
        self.grid = np.ones((2 * self.env_size+1, 2 * self.env_size + 1))
        self.grid[self.target_x][self.target_y] = 2

        chaser_y = chaser_pos[1] + self.env_size
        chaser_x = self.env_size - chaser_pos[0]

        self.grid[chaser_x][chaser_y] = 0

    def __call__(self):
        print(self.grid)

# Environment parameters
env_size = 5
state_space_1D = range(-env_size, env_size + 1)
state_space_bounds = ((-env_size, env_size),)*2
action_space = np.arange(4)

# Uncertainy in selecting action
epsilon = 0.1

def trainer():
    # Initial state
    start_state = ((-5, -5), (0, 0))
    
    # Training parameters
    alpha = 0.95
    gamma = 0.95
    max_training_steps = 1000
    max_episode_steps = 25
    update_q = src.update_q.UpdateQ(alpha, gamma)
    
    # Initialize Q-table randomly.
    Q_table = {((x1, y1), (x2, y2)):{a: np.random.random() for a in action_space} \
                for x1 in state_space_1D for y1 in state_space_1D \
                for x2 in [start_state[1][0]] for y2 in [start_state[1][1]]}
                #for x2 in state_space_1D for y2 in state_space_1D}
    
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
            next_state, reward, done = env(action)
            # update Q-table
            Q_table = update_q(Q_table, state, action, next_state, reward)
            # check if agent has reach target
            if done:
            #if reward == src.reward.GOAL_REWARD:
                #print(state, action, next_state, reward, training_step, episode_step)
                break
            state = next_state
            episode_step += 1
        training_step += 1
    return Q_table   

def render(state, env_size):
    rendering = Render(state[0], state[1], env_size)
    time.sleep(0.5)
    rendering()

def runner(Q_table):
    # Initial state
    start_state = ((-5, -5), (0, 0))
    
    # Running parameters
    max_running_steps = 25
    
    # Intialize environment
    env = src.env.Env(state_space_bounds,
                      action_space,
                      src.reward.two_agent_chasing_reward_nd_grid,
                      src.transition.transition_2d_grid,
                      start_state)
    state = start_state
    render(state, env_size)
    for i in range(0, max_running_steps):
        action = src.select_action.select_action(state, Q_table, epsilon)
        next_state, reward, done = env(action)
        state = next_state
        print(i)
        render(state, env_size)
        if done:
            print(i+1)
            break

def main():
    Q_table = trainer()
    runner(Q_table)

if __name__ == '__main__':
    main()