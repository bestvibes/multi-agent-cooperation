import inspect

from src.reward import TwoAgentChasingRewardNdGridWithObstacles, TwoAgentChaseeRewardNdGridWithObstacles
from src.transition import Transition2dGridSingleAgentCardinal, Transition2dGridMultiAgentCardinal
from src.rendering import Render2DGrid
from src.util import save_nn
from src.util_types import ActionCardinal

from src.single_agent_trainer import TrainSingleAgentPolicy, RenderSingleAgentPolicy
from src.multi_agent_trainer import TrainMultiAgentPolicies, RenderMultiAgentPolicies

from src.dummy_agent import TrainDummyAgent, DummyCardinalStationaryPolicy, DummyCardinalRandomPolicy, zero_reward

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

    train_policy = TrainSingleAgentPolicy(transition, reward, algorithm)

    trained_q_table = train_policy(start_state, done_chasing, max_training_steps=2000, max_episode_steps=100)
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

    train_policy = TrainMultiAgentPolicies(transition, [reward1, zero_reward], [algorithm1, algorithm2])

    trained_q_table, chasee_policy = train_policy(start_state, done_chasing, max_training_steps=10000, max_episode_steps=100)
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
    policy_net = DQN()
    algorithm = TrainDQN(policy_net)#, plot_filename=f"plot_{inspect.currentframe().f_code.co_name}.png")

    train_policy = TrainSingleAgentPolicy(transition, reward, algorithm)

    trained_policy_net = train_policy(start_state, done_chasing, max_training_steps=1000, max_episode_steps=20)
    save_nn(trained_policy_net, "dqn.st")
    policy = DQNPolicy(trained_policy_net, epsilon=0)

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
    policy_net = DQN()
    algorithm1 = TrainDQN(policy_net, plot_filename=f"plot_{inspect.currentframe().f_code.co_name}_1.png")
    algorithm2 = TrainDummyAgent(DummyCardinalStationaryPolicy())

    f_train = TrainMultiAgentPolicies(transition, [reward1, reward2], [algorithm1, algorithm2])

    trained_policy_net, chasee_policy = f_train(start_state, done_chasing, max_training_steps=100, max_episode_steps=100)
    chaser_policy = DQNPolicy(trained_policy_net, epsilon=0)

    renderer = Render2DGrid(obstacles, env_size)

    runner = RenderMultiAgentPolicies(renderer, [chaser_policy, chasee_policy], transition, done_chasing)
    runner(start_state)

if __name__ == '__main__':
    dqn_multi()
