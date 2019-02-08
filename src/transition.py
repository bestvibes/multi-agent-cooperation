import sys
from itertools import starmap

from src.util_types import ActionCardinal

class Transition2dGridSingleAgentCardinal(object):
    def __init__(self, state_space: tuple):
        self.grid_x_bounds = state_space[0]
        self.grid_y_bounds = state_space[1]

    def __call__(self, state: tuple, action) -> tuple:
        action = ActionCardinal(action) # assume action space is ActionCardinal

        new_state = _transition_actor_cardinal(self.grid_x_bounds, self.grid_y_bounds, state[0], action)

        return (new_state,) + state[1:]

class Transition2dGridMultiAgentCardinal(object):
    def __init__(self, state_space: tuple):
        self.grid_x_bounds = state_space[0]
        self.grid_y_bounds = state_space[1]

    def __call__(self, state: tuple, actions: list) -> tuple:
        current_chaser_state = state[0]
        current_chasee_state = state[1]

        actions = list(map(ActionCardinal, actions)) # assume action space is ActionCardinal

        return tuple(starmap(lambda s, a: _transition_actor_cardinal(self.grid_x_bounds, self.grid_y_bounds, s, a), zip(state, actions)))

def _transition_actor_cardinal(grid_x_bounds: tuple,
                                grid_y_bounds: tuple,
                                actor_state: tuple,
                                actor_action: ActionCardinal) -> tuple:
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
