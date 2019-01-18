from collections import namedtuple

# Action space
ACTION_UP = 0
ACTION_DOWN = 1    
ACTION_LEFT = 2
ACTION_RIGHT = 3

# A single transition structure
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))