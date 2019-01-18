import numpy as np
import copy
import time

import src.select_action
import src.update_q
import src.reward
import src.transition

EPSILON = 0.1

def init_random_q_table(action_space: list, state_space_1D: list, start_state: tuple):
	# Initialize Q-table randomly.
    return {((x1, y1), (x2, y2)):{a: np.random.random() for a in action_space} \
            for x1 in state_space_1D for y1 in state_space_1D \
            for x2 in [start_state[1][0]] for y2 in [start_state[1][1]]}
            #for x2 in state_space_1D for y2 in state_space_1D}

def q_learning_trainer(Q_table: dict,
            initial_env,
            start_state: tuple,
            alpha: float=0.95,
            gamma: float=0.95,
            max_training_steps: int=1000,
            max_episode_steps: int=25,
            epsilon: float=EPSILON):

    update_q = src.update_q.UpdateQ(alpha, gamma)

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

def q_learning_runner(Q_table: dict,
            env,
            start_state: tuple,
            renderer: callable,
            max_running_steps: int=25,
            render_interval: float=0.5,
            epsilon: float=EPSILON):
    state = start_state

    renderer(state)
    for i in range(0, max_running_steps):
        action = src.select_action.select_action(state, Q_table, epsilon)
        next_state, reward, done = env(action)
        state = next_state
        print(i)
        renderer(state)
        time.sleep(render_interval)
        if done: break
