# rendering continuous env
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
from matplotlib import pyplot as plt
from matplotlib import animation

class Render_continuous():
    def __init__(self, state_space : tuple, obstacles: list):
        self.xbound, self.ybound = state_space
        self.obstacles = obstacles
    
    def __call__(self, state_sequence, reward_sequence):
        fig = plt.figure()
        ax = plt.axes(xlim=(-1, self.xbound+1), ylim=(-1, self.ybound+1))
        chaser, chasee = ax.plot([], [], 'bo', [], [], 'ro')
        
        def init():
            chaser.set_data([], [])
            chasee.set_data([], [])
            return chaser, chasee
        
        def animate(i):
            state = state_sequence[i]
            chaser.set_data([state[0][0][0]],[state[0][0][1]])
            chasee.set_data([state[1][0][0]],[state[1][0][1]])
            return chaser, chasee        
        
        frame_number = len(state_sequence)
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=frame_number, interval=200, blit=True)
        return anim