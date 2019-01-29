"""
transition_continous
input: state_space, state, action
    state_space: (x bound, y bound)
    state: ((chaser coord, chaser v), (chasee coord, chasee v))
    action: (delta v on x axis, delta v on y axis)
output: next_state
    next_state: ((new chaser coord, new chaser v), (new chasee coord, new chasee v))
Return the next_state to which action will lead from current state.
In case of out of bound action, stays on the bounded (no reflection)
"""
def make_inbound(coord, xbound, ybound):
    x, y = coord
    newx = max(min(x, xbound), 0)
    newy = max(min(y, ybound), 0)
    return newx, newy

def transition_continuous(state_space: tuple, state: tuple, action: int) -> tuple:
    xbound, ybound = state_space
    chaser_coord, chaser_v = state[0]
    chasee_coord, chasee_v = state[1]
    
    #update chasee state
    new_chasee_coord = tuple(map(sum, zip(chasee_coord, chasee_v)))
    new_chasee_state = (make_inbound(new_chasee_coord,xbound,ybound), chasee_v)
    
    #update chaser state
    new_chaser_v = tuple(map(sum, zip(chaser_v, action)))
    new_chaser_coord = tuple(map(sum, zip(chaser_coord, new_chaser_v)))
    new_chaser_state = (make_inbound(new_chaser_coord,xbound,ybound), new_chaser_v)
    
    return (new_chaser_state, new_chasee_state)

