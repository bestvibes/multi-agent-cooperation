import numpy as np
import time
import copy

import src.select_action
import src.env
import src.update_q
import src.reward
import src.transition
import src.rendering

# Environment parameters
env_size = 5
state_space_1D = range(-env_size, env_size + 1)
state_space_bounds = ((-env_size, env_size),)*2
action_space = np.arange(4)
obstacles = [(-3,-4), (-1,0), (0,-1)]
start_state = ((-5, -5), (0, 0))

EPSILON = 0.1

def done_chasing(state: tuple) -> bool:
    chaser_state = state[0]
    chasee_state = state[1]
    return chaser_state == chasee_state

def trainer(initial_env,
            start_state: tuple,
            alpha: float=0.95,
            gamma: float=0.95,
            max_training_steps: int=1000,
            max_episode_steps: int=25,
            epsilon: float=EPSILON):

    update_q = src.update_q.UpdateQ(alpha, gamma)
    
    # Initialize Q-table randomly.
    Q_table = {((x1, y1), (x2, y2)):{a: np.random.random() for a in action_space} \
                for x1 in state_space_1D for y1 in state_space_1D \
                for x2 in [start_state[1][0]] for y2 in [start_state[1][1]]}
                #for x2 in state_space_1D for y2 in state_space_1D}

    # Training
    training_step = 0
    while(training_step <= max_training_steps):
        # Initialize environment
        env = copy.deepcopy(initial_env)
        state = start_state

        episode_step = 0
        while(episode_step <= max_episode_steps):
            action = src.select_action.select_action(state, Q_table, epsilon)
            next_state, reward, done = env(action)
            Q_table = update_q(Q_table, state, action, next_state, reward)
            if done: break

            state = next_state
            episode_step += 1
        training_step += 1
    return Q_table   

def render(state, obstacles, env_size):
    rendering = src.rendering.Render(state[0], state[1], obstacles, env_size)
    time.sleep(0.5)
    rendering()

def runner(Q_table, env, start_state: tuple, max_running_steps: int=25, epsilon: float=EPSILON):
    state = start_state

    render(state, obstacles, env_size)
    for i in range(0, max_running_steps):
        action = src.select_action.select_action(state, Q_table, epsilon)
        next_state, reward, done = env(action)
        state = next_state
        print(i)
        render(state, obstacles, env_size)
        if done: break

def main():
    env = src.env.Env(state_space_bounds,
                      action_space,
                      src.reward.TwoAgentChasingRewardNdGridWithObstacles(obstacles),
                      src.transition.transition_2d_grid,
                      done_chasing,
                      start_state,
                      obstacles)

    Q_table = trainer(env, start_state)
    runner(Q_table, env, start_state)

if __name__ == '__main__':
    main()