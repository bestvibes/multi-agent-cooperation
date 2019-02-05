from numpy.random import random as np_random

def init_random_q_table_2d_square_2_agents(action_space, state_space_1D: list):
    # Initialize Q-table randomly.
    return {((x1, y1), (x2, y2)):{a: np_random() for a in action_space} \
            for x1 in state_space_1D for y1 in state_space_1D \
            for x2 in state_space_1D for y2 in state_space_1D}
