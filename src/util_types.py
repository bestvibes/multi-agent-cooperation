from typing import NamedTuple, Any
from enum import Enum

class ActionCardinal(Enum):
    # Action space
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

# A single transition structure
Transition = NamedTuple('Transition', (('state', tuple),
                                       ('action', Any),
                                       ('next_state', tuple),
                                       ('reward', float)))
