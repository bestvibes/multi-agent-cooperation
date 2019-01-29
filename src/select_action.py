from numpy.random import binomial
import random

from src.util_types import ActionCardinal

def select_action(state : tuple,
                  qtable : dict,
                  epsilon : float = 0) -> ActionCardinal:
    if state not in qtable:
        raise KeyError("State not found in Q-table")
    action2qvalue = qtable[state]
    random_choice = binomial(1, epsilon)
    if random_choice:
        keys = list(action2qvalue.keys())
        return keys[random.randrange(0, len(keys))]
    else:
        max_action = list(action2qvalue.keys())[0];
        max_qvalue = action2qvalue[max_action];
        for action, qvalue in action2qvalue.items():
            if qvalue > max_qvalue:
                max_qvalue = qvalue
                max_action = action
        return ActionCardinal(max_action)

def select_action_chasee(action_type,
                            state: tuple,
                            chaser_next_action: ActionCardinal,
                            epsilon: float = 0.1) -> ActionCardinal:
    random_choice = True #binomial(1, epsilon)
    if random_choice:
        return action_type(random.randrange(0, len(action_type)))
    else:
        return chaser_next_action
