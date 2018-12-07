import math

from src.util import get_coordinates_l2_dist

GOAL_REWARD = 1000
OUT_OF_BOUNDS_REWARD = -100

def two_agent_chasing_reward_nd_grid(state: tuple, action: int, next_state: tuple) -> float:
    next_chaser_state = next_state[0]
    next_chasee_state = next_state[1]

    chaser_chasee_l2_dist = get_coordinates_l2_dist(next_chaser_state, next_chasee_state)

    if chaser_chasee_l2_dist == 0:
        return GOAL_REWARD
    else:
        return -chaser_chasee_l2_dist

class TwoAgentChasingRewardNdGridWithObstacles(object):
    def __init__(self, obstacles: list):
        if obstacles and not all(map(lambda o: isinstance(o, tuple), obstacles)):
            raise ValueError("Invalid obstacles format!")

        self.obstacles = obstacles

    def __call__(self, state: tuple, action: int, next_state: tuple) -> float:
        next_chaser_state = next_state[0]
        next_chasee_state = next_state[1]

        chaser_chasee_l2_dist = get_coordinates_l2_dist(next_chaser_state, next_chasee_state)

        if self.obstacles and next_chaser_state in self.obstacles:
            return OUT_OF_BOUNDS_REWARD
        elif chaser_chasee_l2_dist == 0:
            return GOAL_REWARD
        else:
            return -chaser_chasee_l2_dist
