import gym
ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME)

def cartpole_transition_function(state, action):
    pass

def cartpole_reward_function(state, action, next_state):
    # reward = -reward when done
    pass

def cartpole_get_initial_state():
    return list(env.reset())

def cartpole_done_function(state):
    pass