import numpy as np

OBSTACLE = 'O'

class Render2DGrid():
    def __init__(self, obstacles: list, env_size: int):
        self.env_size = env_size
        self.obstacles = obstacles

        self.grid = [["."]*(2 * self.env_size + 1)]*(2 * self.env_size + 1)
        self.grid = np.asarray(self.grid)

    def __call__(self, state: tuple):
        chaser_pos = state[0]
        target_pos = state[1]

        # reset grid
        self.grid.fill('.')

        num_agents = len(state)
        # prioritize showing lower index agents
        for i in reversed(range(num_agents)):
            y = state[i][0] + self.env_size
            x = self.env_size - state[i][1]
            self.grid[x][y] = str(i)
        
        for obs in self.obstacles:
            obs_y = obs[0] + self.env_size
            obs_x = self.env_size - obs[1]
            self.grid[obs_x][obs_y] = OBSTACLE

        print('\n'.join(map(lambda x: ' '.join(x), self.grid)))
        return self.grid
