from collections import namedtuple

# Action space
ACTION_UP = 0
ACTION_DOWN = 1    
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_SPACE = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# A single transition structure
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))