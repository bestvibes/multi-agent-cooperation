from src.reward import TwoAgentChasingRewardNdGridWithObstacles, TwoAgentChaseeRewardNdGridWithObstacles
from src.transition import Transition2dGridSingleAgentCardinal, Transition2dGridMultiAgentCardinal
from src.rendering import Render2DGrid
from src.util_types import ActionCardinal

from src.single_agent_game import SingleAgentGame, SingleAgentRunner
from src.multi_agent_game import MultiAgentGame, MultiAgentRunner

from src.dummy_agent import NoLearningAlgorithm, DummyCardinalStationaryPolicy, DummyCardinalRandomPolicy, zero_reward

# Q Learning
from src.q_learning.algorithm import QLearningAlgorithm
from src.q_learning.util import init_random_q_table_2d_square_2_agents

# DQN
from src.dqn.algorithm import DQNAlgorithm, save_dqn

# Policy Gradient
from src.policy_gradient.algorithm import PolicyGradientAlgorithm

def done_chasing(state: tuple) -> bool:
    chaser_state = state[0]
    chasee_state = state[1]
    return chaser_state == chasee_state

# def main_policy_gradient():
#     model_path = "policy_gradient.st"

#     # Environment parameters
#     env_size = 5
#     state_space_1D = range(-env_size, env_size + 1)
#     state_space_bounds = ((-env_size, env_size),)*2
#     obstacles = [(-3,-4), (-1,0), (0,-1)]
#     start_state = ((-5, -5), (0, 0))

#     env = src.env.Env(state_space_bounds,
#                       ActionCardinal,
#                       src.reward.TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles),
#                       src.transition.transition_2d_grid,
#                       done_chasing,
#                       start_state,
#                       obstacles)

#     policy_gradient_trainer(env, start_state, model_path, max_training_steps=1000, max_episode_steps=50, plot_interval=2)
#     renderer = src.rendering.Render2DGrid(obstacles, env_size)
#     policy_gradient_runner(env, start_state, renderer, model_path, max_running_steps=50)

def q_learning_single():
    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    transition = Transition2dGridSingleAgentCardinal(state_space_bounds)
    reward = TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles)
    initial_q_table = init_random_q_table_2d_square_2_agents(ActionCardinal, state_space_1D)
    trainer = QLearningAlgorithm(initial_q_table)

    game = SingleAgentGame(transition, reward, trainer)

    policy = game(start_state, done_chasing, max_training_steps=2000, max_episode_steps=100)

    renderer = Render2DGrid(obstacles, env_size)

    runner = SingleAgentRunner(policy, renderer, transition, done_chasing)
    runner(start_state)

def q_learning_multi():
    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    transition = Transition2dGridMultiAgentCardinal(state_space_bounds)
    reward1 = TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles)
    initial_q_table = init_random_q_table_2d_square_2_agents(ActionCardinal, state_space_1D)
    trainer1 = QLearningAlgorithm(initial_q_table)
    trainer2 = NoLearningAlgorithm(DummyCardinalRandomPolicy())

    game = MultiAgentGame(transition, [reward1, zero_reward], [trainer1, trainer2])

    policies = game(start_state, done_chasing, max_training_steps=10000, max_episode_steps=100)

    renderer = Render2DGrid(obstacles, env_size)

    runner = MultiAgentRunner(policies, renderer, transition, done_chasing)
    runner(start_state)

def dqn_single():
    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    transition = Transition2dGridSingleAgentCardinal(state_space_bounds)
    reward = TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles)
    trainer = DQNAlgorithm(plot_filename="dqn_plot.png")

    game = SingleAgentGame(transition, reward, trainer)

    policy = game(start_state, done_chasing, max_training_steps=1000, max_episode_steps=50)
    save_dqn(policy, "dqn.st")

    renderer = Render2DGrid(obstacles, env_size)

    runner = SingleAgentRunner(policy, renderer, transition, done_chasing)
    runner(start_state)

def dqn_multi():
    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    transition = Transition2dGridMultiAgentCardinal(state_space_bounds)
    reward1 = TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles)
    reward2 = TwoAgentChaseeRewardNdGridWithObstacles(state_space_bounds, obstacles)
    trainer1 = DQNAlgorithm(plot_filename="dqn_1")
    trainer2 = DQNAlgorithm(plot_filename="dqn_2")

    game = MultiAgentGame(transition, [reward1, reward2], [trainer1, trainer2])

    policies = game(start_state, done_chasing, max_training_steps=1000, max_episode_steps=100)

    renderer = Render2DGrid(obstacles, env_size)

    runner = MultiAgentRunner(policies, renderer, transition, done_chasing)
    runner(start_state)

if __name__ == '__main__':
    dqn_multi()
