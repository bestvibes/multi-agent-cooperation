import math

from src.util import get_coordinates_l2_dist

GOAL_REWARD = 1
OUT_OF_BOUNDS_REWARD = -10
TIME_PENALTY = -1

def two_agent_chasing_reward_nd_grid(state: tuple, action: int, next_state: tuple) -> float:
    next_chaser_state = next_state[0]
    next_chasee_state = next_state[1]

    chaser_chasee_l2_dist = get_coordinates_l2_dist(next_chaser_state, next_chasee_state)

    if chaser_chasee_l2_dist == 0:
        return GOAL_REWARD
    else:
        return TIME_PENALTY

class TwoAgentChasingRewardNdGridWithObstacles(object):
    def __init__(self, state_space: tuple, obstacles: list):
        if obstacles and not all(map(lambda o: isinstance(o, tuple), obstacles)):
            raise ValueError("Invalid obstacles format!")

        self.obstacles = obstacles
        self.grid_x_bounds = state_space[0]
        self.grid_y_bounds = state_space[1]

    def __call__(self, state: tuple, action: int, next_state: tuple) -> float:
        next_chaser_state = next_state[0]
        next_chasee_state = next_state[1]

        chaser_chasee_l2_dist = get_coordinates_l2_dist(next_chaser_state, next_chasee_state)

        # print(self.grid_x_bounds)
        rew = 0
        if self.obstacles and next_chaser_state in self.obstacles:
            # rew =  0
            rew = OUT_OF_BOUNDS_REWARD
        elif next_chaser_state[0] < self.grid_x_bounds[0] or next_chaser_state[0] > self.grid_x_bounds[1]:
            rew = OUT_OF_BOUNDS_REWARD
        elif next_chaser_state[1] < self.grid_y_bounds[0] or next_chaser_state[1] > self.grid_y_bounds[1]:
            rew = OUT_OF_BOUNDS_REWARD
        elif chaser_chasee_l2_dist == 0:
            rew = GOAL_REWARD
        else:
            rew = TIME_PENALTY - chaser_chasee_l2_dist

        # print("step reward", rew)
        return rew
