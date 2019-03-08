def transition_2d_grid(state_space: tuple, state: tuple, action: int) -> tuple:
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    grid_x_bounds = state_space[0]
    grid_y_bounds = state_space[1]

    current_chaser_state = state[0]
    current_chasee_state = state[1]

    if action == ACTION_RIGHT:
        new_chaser_state = (current_chaser_state[0]+1, current_chaser_state[1])
        # if (current_chaser_state[0] + 1) <= grid_x_bounds[1]:
        #     new_chaser_state = (current_chaser_state[0]+1, current_chaser_state[1])
        # else:
        #     new_chaser_state = current_chaser_state
    elif action == ACTION_LEFT:
        new_chaser_state = (current_chaser_state[0]-1, current_chaser_state[1])
        # if (current_chaser_state[0] - 1) >= grid_x_bounds[0]:
        #     new_chaser_state = (current_chaser_state[0]-1, current_chaser_state[1])
        # else:
        #     new_chaser_state = current_chaser_state
    elif action == ACTION_UP:
        new_chaser_state = (current_chaser_state[0], current_chaser_state[1]+1)
        # if (current_chaser_state[1] + 1) <= grid_y_bounds[1]:
        #     new_chaser_state = (current_chaser_state[0], current_chaser_state[1]+1)
        # else:
        #     new_chaser_state = current_chaser_state
    elif action == ACTION_DOWN:
        new_chaser_state = (current_chaser_state[0], current_chaser_state[1]-1)
        # if (current_chaser_state[1] - 1) >= grid_y_bounds[0]:
        #     new_chaser_state = (current_chaser_state[0], current_chaser_state[1]-1)
        # else:
        #     new_chaser_state = current_chaser_state

    return (new_chaser_state, current_chasee_state)
