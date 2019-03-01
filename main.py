import copy
import inspect
from itertools import starmap

import torch.nn as nn

from src.reward import TwoAgentChasingRewardNdGridWithObstacles, TwoAgentChaseeRewardNdGridWithObstacles
from src.transition import Transition2dGridSingleAgentCardinal, Transition2dGridMultiAgentCardinal
from src.rendering import Render2DGrid
from src.util import save_nn
from src.util_types import ActionCardinal

from src.single_agent_trainer import RenderSingleAgentPolicy
from src.multi_agent_trainer import RenderMultiAgentPolicies

from src.dummy_agent import TrainDummyAgent, DummyCardinalStationaryPolicy, DummyCardinalRandomPolicy, zero_reward

import src.nn

# Q Learning
from src.q_learning.algorithm import TrainQTable, QLearningPolicy
from src.q_learning.util import init_random_q_table_2d_square_2_agents

# DQN
from src.dqn.dqn import DQN
from src.dqn.algorithm import TrainDQN, DQNPolicy

def done_chasing(state: tuple) -> bool:
    chaser_state = state[0]
    chasee_state = state[1]
    return chaser_state == chasee_state

def q_learning_single():
    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = [(1,0), (-1,0), (0,-1)]
    start_state = ((-5, -5), (0, 0))

    transition = Transition2dGridSingleAgentCardinal(state_space_bounds)
    reward = TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles)
    initial_q_table = init_random_q_table_2d_square_2_agents(ActionCardinal, state_space_1D)
    algorithm = TrainQTable(initial_q_table)

    max_training_steps = 2000
    max_episode_steps = 100

    training_step = 0
    while(training_step <= max_training_steps):
        state = copy.deepcopy(start_state)

        episode_step = 0
        while(episode_step <= max_episode_steps):
            action = algorithm.select_action(state)
            #print(action)
            next_state = transition(state, action)
            step_reward = reward(state, action, next_state)

            algorithm.train_episode_step(state, action, next_state, step_reward)

            if (done_chasing(next_state)): break

            state = next_state
            episode_step += 1

        algorithm.train_training_step()
        training_step += 1
        if (training_step % 50 == 0):
            print(f"training step: {training_step}")

    trained_q_table = algorithm.get_policy()

    policy = QLearningPolicy(trained_q_table, epsilon=0)

    renderer = Render2DGrid(obstacles, env_size)
    runner = RenderSingleAgentPolicy(renderer, policy, transition, done_chasing)
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
    algorithm1 = TrainQTable(initial_q_table)
    algorithm2 = TrainDummyAgent(DummyCardinalRandomPolicy())
    algorithms = [algorithm1, algorithm2]
    rewards = [reward1, zero_reward]

    max_training_steps = 10000
    max_episode_steps = 100

    training_step = 0
    while(training_step <= max_training_steps):
        state = copy.deepcopy(start_state)

        episode_step = 0
        while(episode_step <= max_episode_steps):
            actions = list(map(lambda t: t.select_action(state), algorithms))
            next_state = transition(state, actions)
            step_rewards = list(starmap(lambda rf, a: rf(state, a, next_state), zip(rewards, actions)))

            list(starmap(lambda t, a, r: t.train_episode_step(state, a, next_state, r), zip(algorithms, actions, step_rewards)))

            if (done_chasing(next_state)): break

            state = next_state
            episode_step += 1

        list(map(lambda t: t.train_training_step(), algorithms))
        training_step += 1
        if (training_step % 50 == 0):
            print(f"training step: {training_step}")

    trained_q_table, chasee_policy = list(map(lambda t: t.get_policy(), algorithms))

    chaser_policy = QLearningPolicy(trained_q_table, epsilon=0)

    renderer = Render2DGrid(obstacles, env_size)
    runner = RenderMultiAgentPolicies(renderer, [chaser_policy, chasee_policy], transition, done_chasing)
    runner(start_state)

def dqn_single():
    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = []
    start_state = ((-5, -5), (5, 5))

    transition = Transition2dGridSingleAgentCardinal(state_space_bounds)
    reward = TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles)
    policy_model = src.nn.ReluSoftmaxNN([nn.functional.linear,
                                         nn.functional.linear,
                                         nn.functional.linear,
                                         nn.functional.linear])
    
    parameters = [*src.nn.create_init_linear_layer_weights_and_bias(4, 8),
                    *src.nn.create_init_linear_layer_weights_and_bias(8, 8),
                    *src.nn.create_init_linear_layer_weights_and_bias(8, 8),
                    *src.nn.create_init_linear_layer_weights_and_bias(8, 4)]
    algorithm = TrainDQN(policy_model, parameters)#, plot_filename=f"plot_{inspect.currentframe().f_code.co_name}.png")

    max_training_steps = 200
    max_episode_steps = 20

    training_step = 0
    while(training_step <= max_training_steps):
        state = copy.deepcopy(start_state)

        episode_step = 0
        while(episode_step <= max_episode_steps):
            action = algorithm.select_action(state)
            #print(action)
            next_state = transition(state, action)
            step_reward = reward(state, action, next_state)

            algorithm.train_episode_step(state, action, next_state, step_reward)

            if (done_chasing(next_state)): break

            state = next_state
            episode_step += 1

        algorithm.train_training_step()
        training_step += 1
        if (training_step % 50 == 0):
            print(f"training step: {training_step}")

    trained_policy_parameters = algorithm.get_policy()

    policy = DQNPolicy(policy_model, trained_policy_parameters, epsilon=0)

    renderer = Render2DGrid(obstacles, env_size)
    runner = RenderSingleAgentPolicy(renderer, policy, transition, done_chasing)
    runner(start_state)

def dqn_multi():
    # Environment parameters
    env_size = 5
    state_space_1D = range(-env_size, env_size + 1)
    state_space_bounds = ((-env_size, env_size),)*2
    obstacles = [(-3,-4), (-1,0), (0,-1)]
    start_state = ((-5, -5), (5, 5))

    transition = Transition2dGridMultiAgentCardinal(state_space_bounds)
    reward1 = TwoAgentChasingRewardNdGridWithObstacles(state_space_bounds, obstacles)
    reward2 = TwoAgentChaseeRewardNdGridWithObstacles(state_space_bounds, obstacles)

    policy_model = src.nn.ReluSoftmaxNN([nn.functional.linear,
                                         nn.functional.linear,
                                         nn.functional.linear,
                                         nn.functional.linear])
    
    parameters = [*src.nn.create_init_linear_layer_weights_and_bias(4, 8),
                    *src.nn.create_init_linear_layer_weights_and_bias(8, 8),
                    *src.nn.create_init_linear_layer_weights_and_bias(8, 8),
                    *src.nn.create_init_linear_layer_weights_and_bias(8, 4)]

    algorithm1 = TrainDQN(policy_model, parameters, plot_filename=f"plot_{inspect.currentframe().f_code.co_name}_1.png")
    algorithm2 = TrainDummyAgent(DummyCardinalStationaryPolicy())
    algorithms = [algorithm1, algorithm2]
    rewards = [reward1, reward2]

    max_training_steps = 100
    max_episode_steps = 100

    training_step = 0
    while(training_step <= max_training_steps):
        state = copy.deepcopy(start_state)

        episode_step = 0
        while(episode_step <= max_episode_steps):
            actions = list(map(lambda t: t.select_action(state), algorithms))
            next_state = transition(state, actions)
            step_rewards = list(starmap(lambda rf, a: rf(state, a, next_state), zip(rewards, actions)))

            list(starmap(lambda t, a, r: t.train_episode_step(state, a, next_state, r), zip(algorithms, actions, step_rewards)))

            if (done_chasing(next_state)): break

            state = next_state
            episode_step += 1

        list(map(lambda t: t.train_training_step(), algorithms))
        training_step += 1
        if (training_step % 50 == 0):
            print(f"training step: {training_step}")

    trained_policy_parameters, chasee_policy = list(map(lambda t: t.get_policy(), algorithms))

    chaser_policy = DQNPolicy(policy_model, trained_policy_parameters, epsilon=0)

    renderer = Render2DGrid(obstacles, env_size)
    runner = RenderMultiAgentPolicies(renderer, [chaser_policy, chasee_policy], transition, done_chasing)
    runner(start_state)

if __name__ == '__main__':
    dqn_multi()
