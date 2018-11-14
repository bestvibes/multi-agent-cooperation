import math

GOAL_REWARD = 1000
OUT_OF_BOUNDS_REWARD = -1000

def two_agent_chasing_reward_nd_grid(state: tuple, action: int, next_state: tuple) -> float:
    next_chaser_state = next_state[0]
    next_chasee_state = next_state[1]

    num_dims = len(next_chaser_state)
    dim_dists = map(lambda i: next_chaser_state[i] - next_chasee_state[i], range(num_dims))

    chaser_chasee_l2_dist = math.sqrt(sum(map(lambda d: d**2, dim_dists)))

    if state == next_state: # if we stepped out of bounds
        return OUT_OF_BOUNDS_REWARD
    elif chaser_chasee_l2_dist == 0:
        return GOAL_REWARD
    else:
        return -chaser_chasee_l2_dist
