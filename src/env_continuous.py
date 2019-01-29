import math
"""
Env_continuous class
Initialize:
    state_space: (x bound, y bound)
    action_space: max magnitude of delta v ???
    reward: reward function
    transition: transition function
    done: a function that returns true if terminal state occurs
    start_state: ((chaser coord, chaser v), (chasee coord, chasee v))
    obstacles: unfinished feature
Call:
    input:
        action: delta v = (delta v_x, delta v_y)
    output:
        [action is capped by action_space]
        next_state: as given by transition function
        reward: as given by reward function
        done_condition: as given by done function
"""
class Env_continuous(object):
    def __init__(self, state_space: tuple,
                       action_space: float,
                       reward: callable,
                       transition: callable,
                       done: callable,
                       start_state: tuple,
                       obstacles: list):
        # check action_space
        if action_space <= 0:
            raise ValueError("env: max magnitude of action is non-positive!")
        # check dimension of start state
        if not all(map(lambda x: len(x) == 2, start_state[0]+start_state[1])):
            raise ValueError("env: start state dimension incorrect!")
        # check if start state is in bound
        for coords, v in start_state:
            for dim, coord in enumerate(coords):
                if coord < 0 or coord > state_space[dim]:
                    raise ValueError("env: start state out of state space bounds!")
        
        self.state_space = state_space
        self.action_space = action_space
        self.reward = reward
        self.transition = transition
        self.done = done
        self.current_state = start_state
        self.obstacles = obstacles

    def cap_action(self, action):
        action_magnitude = math.sqrt(sum(map(lambda d: d**2, action)))
        if action_magnitude > self.action_space:
            action = tuple(map(lambda x: x/action_magnitude*self.action_space, action))
        return action
        
    def __call__(self, action: tuple) -> (tuple, float, bool):
        action = self.cap_action(action)
        next_state = self.transition(self.state_space, self.current_state, action)
        reward = self.reward(self.current_state, action, next_state)
        self.current_state = next_state
        return (next_state, reward, self.done(next_state))