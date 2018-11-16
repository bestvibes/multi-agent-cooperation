import numpy as np

def main():
    # Initialize environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    action_space = np.arange(4)

    # Initialize training parameters
    alpha = 0.95
    gamma = 0.95
    max_training_steps = 1000
    max_episode_steps = 25

    # Initialize Q-table randomly.
    Q_table = {((x1, y1), (x2, y2)): {a: random.random() for a in action_space} for x1 in state_space_1D for y1 in state_space_1D for x2 in state_space_1D for y2 in state_space_1D}
    
    # Initialize state
    state = ((0, 0), (0, 0))

    # Training
    training_step = 0
    while(training_step <= max_training_steps):
        episode_step = 0
        while(episode_step <= max_episode_steps):
            # select action
            action = select_action(Q_table, state)

            # update environment
            next_state, reward = step(action)

            # update Q-table
            Q_table = UpdateQ(Q_table, state, action, next_state, reward)

            episode_step += 1

        training_step += 1



if __name__ == '__main__':
    main()