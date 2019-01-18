from src.util import get_coordinates_l2_dist
"""
reward_continuous
def two_agent_chasing_reward_nd_grid(state: tuple, action: int, next_state: tuple) -> float:
input: state, action, next_state
output: reward
    reward is the negative l2 distance between the agent and the target
"""
GOAL_REWARD = 1000

def reward_continuous(state, action, next_state):
    next_chaser_state = next_state[0][0]
    next_chasee_state = next_state[1][0]
    chaser_chasee_l2_dist = get_coordinates_l2_dist(next_chaser_state, next_chasee_state)

    if chaser_chasee_l2_dist == 0:
        return GOAL_REWARD
    else:
        return -chaser_chasee_l2_dist