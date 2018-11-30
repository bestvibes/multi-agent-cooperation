import numpy as np

import src.env
import src.rendering
import src.q_learning
from src.dqn.main import dqn_trainer, dqn_runner

def done_chasing(state: tuple) -> bool:
    chaser_state = state[0]
    chasee_state = state[1]
    return chaser_state == chasee_state

def main_q_learning():
    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    action_space = np.arange(4)
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    Q_table = src.q_learning.init_random_q_table(action_space, state_space_1D, start_state)

    env = src.env.Env(state_space_bounds,
                      action_space,
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(obstacles),
                      src.transition.transition_2d_grid,
                      done_chasing,
                      start_state,
                      obstacles)

    Q_table = src.q_learning.q_learning_trainer(Q_table, env, start_state)
    renderer = src.rendering.Render2DGrid(obstacles, env_size)
    src.q_learning.q_learning_runner(Q_table, env, start_state, renderer)

def main_dqn():
    model_path = "dqn.st"

    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    action_space = np.arange(4)
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    env = src.env.Env(state_space_bounds,
                      action_space,
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(obstacles),
                      src.transition.transition_2d_grid,
                      done_chasing,
                      start_state,
                      obstacles)

    dqn_trainer(env, start_state, model_path, batch_size=1, max_training_steps=500, max_episode_steps=100, epsilon=0.05)
    renderer = src.rendering.Render2DGrid(obstacles, env_size)
    dqn_runner(env, start_state, renderer, model_path)

if __name__ == '__main__':
    main_dqn()
