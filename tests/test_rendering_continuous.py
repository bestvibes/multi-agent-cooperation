from src.rendering_continuous import Render_continuous
"""
Render_continuous:
    initialize:
        state_space: (x bound, y bound)
        obstacles : unfinished feature
    Call:
        state_sequence: [S1, S2, ... SN]
            Sn = ((chaser coord, chaser velocity),(chasee coord, chasee velocity))
        reward_sequence: [R1, R2, .. R(N-1)]
"""
import numpy as np
from src.env_continuous import Env_continuous
from src.reward_continuous import reward_continuous
from src.transition_continuous import transition_continuous

def straight_to_target(state_space, start_state):
    action_space = 1
    obstacles = None

    def done_chasing(state: tuple) -> bool:
        chaser_state = state[0][0]
        chasee_state = state[1][0]
        return chaser_state == chasee_state

    env = Env_continuous(state_space,
                         action_space,
                         reward_continuous,
                         transition_continuous,
                         done_chasing,
                         start_state,
                         obstacles)
    
    state_sequence = [start_state]
    reward_sequence = []
    state = start_state
    done_cond = False
    action = tuple(np.subtract(state[1][0], state[0][0]))
    while not done_cond:
        next_state, reward, done_cond = env(action)
        action = (0,0)
        state_sequence.append(next_state)
        reward_sequence.append(reward)
        if next_state == state:
            print("Stuck...")
            break
        else:
            state = next_state
    
    return state_sequence, reward_sequence

def test_render():
    start_state = (((3,4),(0,0)),((15,20),(0,0)))
    state_space = (20, 20)
    state_sequence, reward_sequence = straight_to_target(state_space, start_state)
    render = Render_continuous(state_space, None)
    return render(state_sequence, reward_sequence)

if __name__ == '__main__':
    anim = test_render()