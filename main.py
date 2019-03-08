import numpy as np
import random

import src.env
import src.rendering
import src.q_learning
from src.dqn.main import dqn_trainer, dqn_runner
from src.policy_gradient.main import policy_gradient_trainer, policy_gradient_runner
from src.control_with_dqn.main import dqn_runner_control, dqn_trainer_control
from src.a2c.main import a2c_trainer, a2c_runner

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
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
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

    random_start_state = False

    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    action_space = np.arange(4)
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    # obstacles = None
    if random_start_state:
        start_state = ((random.randint(-5,5), random.randint(-5,5)),(0, 0))
    else:
        start_state = ((-5, -5), (0, 0))
    # start_state = ((random.randint(-5,5), random.randint(-5,5)),(0, 0))

    env = src.env.Env(state_space_bounds,
                      action_space,
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
                      src.transition.transition_2d_grid,
                      done_chasing,
                      start_state,
                      obstacles)

    policy_gradient_trainer(env, start_state, model_path, max_training_steps=1000, max_episode_steps=50, plot_interval=0.01, random_start_state=random_start_state)
    renderer = src.rendering.Render2DGrid(obstacles, env_size)
    policy_gradient_runner(env, start_state, renderer, model_path, max_running_steps=30)

def main_dqn_control():
    model_path = "dqn_control.st"

    # Environment parameters
    env_size = 3
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    action_space = np.arange(4)
    obstacles = [(-3,-2), (-1,0), (0,-1)]
    # obstacles = [(0,-1), (-1, 0), (1, 0)]
    # obstacles = None
    # start_state = ((-5, -5), (0, 0))
    start_state = ((random.randint(-1*env_size,env_size), random.randint(-1*env_size,env_size)),(0,0))# (random.randint(-5,5), random.randint(-5,5)))
    # print("start state", start_state)
    # start_state_list = [-5, -5, 0, 0]

    env = src.env.Env(state_space_bounds,
                      action_space,
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
                      src.transition.transition_2d_grid,
                      done_chasing,
                      start_state,
                      obstacles)

    renderer = src.rendering.Render2DGrid(obstacles, env_size)
    dqn_trainer_control(renderer, env, start_state, model_path, max_training_steps=2000, max_episode_steps=50, plot_interval=2)
    dqn_runner_control(env, start_state, renderer, model_path, max_running_steps=50)


def main_a2c():
    value_path = "dqn.st"
    policy_path = "policy.st"

    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    action_space = np.arange(4)
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))
    # start_state = ((random.randint(-5,5), random.randint(-5,5)), (random.randint(-5,5), random.randint(-5,5)))
    # start_state_list = [-5, -5, 0, 0]

    env = src.env.Env(state_space_bounds,
                      action_space,
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
                      src.transition.transition_2d_grid,
                      done_chasing,
                      start_state,
                      obstacles)

    a2c_trainer(env, start_state, save_path_policy_net=policy_path, save_path_value_net=value_path, max_training_steps=10, max_episode_steps=50, plot_interval=2)
    renderer = src.rendering.Render2DGrid(obstacles, env_size)
    a2c_runner(env, start_state, renderer, policy_load_path=policy_path, value_load_path=value_path, max_running_steps=50)
if __name__ == '__main__':
    main_policy_gradient()
