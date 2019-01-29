import sys

from src.util_types import ActionChaserChasee, ActionCardinal

def transition_2d_grid(state_space: tuple, state: tuple, action: ActionChaserChasee) -> tuple:
    grid_x_bounds = state_space[0]
    grid_y_bounds = state_space[1]

    current_chaser_state = state[0]
    current_chasee_state = state[1]

    new_chaser_state = _transition_actor(state_space, current_chaser_state, action.chaser)
    new_chasee_state = _transition_actor(state_space, current_chasee_state, action.chasee)

    return (new_chaser_state, new_chasee_state)

def _transition_actor(state_space: tuple, actor_state: tuple, actor_action: ActionCardinal) -> tuple:
    grid_x_bounds = state_space[0]
    grid_y_bounds = state_space[1]

    if actor_action == ActionCardinal.RIGHT:
        if (actor_state[0] + 1) <= grid_x_bounds[1]:
            new_actor_state = (actor_state[0]+1, actor_state[1])
        else:
            new_actor_state = actor_state
    elif actor_action == ActionCardinal.LEFT:
        if (actor_state[0] - 1) >= grid_x_bounds[0]:
            new_actor_state = (actor_state[0]-1, actor_state[1])
        else:
            new_actor_state = actor_state
    elif actor_action == ActionCardinal.UP:
        if (actor_state[1] + 1) <= grid_y_bounds[1]:
            new_actor_state = (actor_state[0], actor_state[1]+1)
        else:
            new_actor_state = actor_state
    elif actor_action == ActionCardinal.DOWN:
        if (actor_state[1] - 1) >= grid_y_bounds[0]:
            new_actor_state = (actor_state[0], actor_state[1]-1)
        else:
            new_actor_state = actor_state
    elif actor_action == ActionCardinal.STAY:
        new_actor_state = actor_state
    else:
        print("ERROR! invalid action given to transition function: " + str(actor_action))
        sys.exit(1)

    return new_actor_state
