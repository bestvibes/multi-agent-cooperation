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

from src.interpolate import interpolate_parametric

class Render_continuous():
    def __init__(self, state_space : tuple, obstacles: list):
        self.xbound, self.ybound = state_space
        self.obstacles = obstacles
    
    def __call__(self, state_sequence, reward_sequence, show=True):
        fig = plt.figure()
        ax = plt.axes(xlim=(-1, self.xbound+1), ylim=(-1, self.ybound+1))
        chaser, chasee = ax.plot([], [], 'bo', [], [], 'ro')

        # interpolate the trajectories by splining

        t = list(range(0, len(state_sequence)))

        chaser_x = list(map(lambda state: state[0][0][0], state_sequence))
        chaser_y = list(map(lambda state: state[0][0][1], state_sequence))

        chasee_x = list(map(lambda state: state[1][0][0], state_sequence))
        chasee_y = list(map(lambda state: state[1][0][1], state_sequence))

        _, chaser_x, chaser_y = interpolate_parametric(t, chaser_x, chaser_y)
        t, chasee_x, chasee_y = interpolate_parametric(t, chasee_x, chasee_y)
        
        def init():
            chaser.set_data([], [])
            chasee.set_data([], [])
            return chaser, chasee
        
        def animate(i):
            chaser.set_data([chaser_x[i], chaser_y[i]])
            chasee.set_data([chasee_x[i], chasee_y[i]])
            return chaser, chasee        
        
        frame_number = len(t)
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=frame_number, interval=5, blit=True)
        if show:
            plt.show()
        return anim