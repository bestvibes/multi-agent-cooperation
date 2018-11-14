from numpy.random import binomial
import random

def select_action(state, qtable, epsilon=0):
    if state not in qtable:
        return -1
    action2qvalue = qtable[state]
    random_choice = binomial(1, epsilon)
    if random_choice:
        keys = action2qvalue.keys()
        return random.randrange(0, len(keys))
    else:
        max_action = list(action2qvalue.keys())[0];
        max_qvalue = action2qvalue[max_action];
        for action, qvalue in action2qvalue.items():
            if qvalue > max_qvalue:
                max_qvalue = qvalue
                max_action = action
        return max_action