from typing import NamedTuple
from enum import Enum

class ActionCardinal(Enum):
    # Action space
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

ActionChaserChasee = NamedTuple('Action', (('chaser', ActionCardinal), ('chasee', ActionCardinal)))

# A single transition structure
Transition = NamedTuple('Transition', (('state', tuple),
                                       ('action', ActionChaserChasee),
                                       ('next_state', tuple),
                                       ('reward', float)))
