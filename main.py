import numpy as np

import src.env
import src.rendering
import src.q_learning
from src.dqn.main import dqn_trainer, dqn_runner
from src.policy_gradient.main import policy_gradient_trainer, policy_gradient_runner
from src.util_types import ActionCardinal

def done_chasing(state: tuple) -> bool:
    chaser_state = state[0]
    chasee_state = state[1]
    return chaser_state == chasee_state

def main_q_learning():
    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    Q_table = src.q_learning.init_random_q_table(ActionCardinal, state_space_1D, start_state)

    env = src.env.Env(state_space_bounds,
                      ActionCardinal,
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
                      src.transition.transition_2d_grid,
                      done_chasing,
                      start_state,
                      obstacles)

    Q_table = src.q_learning.q_learning_trainer(Q_table, env, start_state, max_training_steps=2000, max_episode_steps=100)
    renderer = src.rendering.Render2DGrid(obstacles, env_size)
    src.q_learning.q_learning_runner(Q_table, env, start_state, renderer, max_running_steps=100)

def main_dqn():
    model_path = "dqn.st"

    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    env = src.env.Env(state_space_bounds,
                      ActionCardinal,
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
                      src.transition.transition_2d_grid,
                      done_chasing,
                      start_state,
                      obstacles)

    dqn_trainer(env, start_state, model_path, batch_size=1, max_training_steps=1000, max_episode_steps=50, epsilon=0.05)
    renderer = src.rendering.Render2DGrid(obstacles, env_size)
    dqn_runner(env, start_state, renderer, model_path)

def main_policy_gradient():
    model_path = "policy_gradient.st"

    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    env = src.env.Env(state_space_bounds,
                      ActionCardinal,
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
                      src.transition.transition_2d_grid,
                      done_chasing,
                      start_state,
                      obstacles)

    policy_gradient_trainer(env, start_state, model_path, max_training_steps=1000, max_episode_steps=50, plot_interval=2)
    renderer = src.rendering.Render2DGrid(obstacles, env_size)
    policy_gradient_runner(env, start_state, renderer, model_path, max_running_steps=50)

if __name__ == '__main__':
    main_policy_gradient()
