from collections.abc import Hashable

class Env(object):
    def __init__(self, state_space: tuple,
                        action_space: list,
                        reward: callable,
                        transition: callable,
                        start_state: tuple):

        if not all(map(lambda a: isinstance(a, Hashable), action_space)):
            raise ValueError("env: all actions are not hashable!")

        if not all(map(lambda x: len(x) == len(state_space), start_state)):
            raise ValueError("env: start state/state space dimension mismatch!")

        for agent_pos in start_state:
            for dim, dim_pos in enumerate(agent_pos):
                if dim_pos <= state_space[dim][0] or dim_pos >= state_space[dim][1]:
                    raise ValueError("env: start state out of state space bounds!")

        self.state_space = state_space
        self.action_space = action_space
        self.reward = reward
        self.transition = transition
        self.current_state = start_state

    def __call__(self, action: int) -> (tuple, float):
        next_state = self.transition(self.current_state, action)
        reward = self.reward(self.current_state, action, next_state)

        self.current_state = next_state

        return (next_state, reward)
