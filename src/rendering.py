import numpy as np

class Render():
    def __init__(self, chaser_pos, target_pos, obstacles, env_size):
        self.env_size = env_size
        self.target_y = target_pos[0] + self.env_size
        self.target_x = self.env_size - target_pos[1]
        #self.target_x = self.env_size - target_pos[0]
        
        #self.grid = np.ones((2 * self.env_size+1, 2 * self.env_size + 1))
        self.grid = [["."]*(2 * self.env_size + 1)]*(2 * self.env_size + 1)
        self.grid = np.asarray(self.grid)
        self.grid[self.target_x][self.target_y] = 'T'

        chaser_y = chaser_pos[0] + self.env_size
        chaser_x = self.env_size - chaser_pos[1]
        
        for obs in obstacles:
            obs_y = obs[0] + self.env_size
            obs_x = self.env_size - obs[1]
            self.grid[obs_x][obs_y] = 'O'

        self.grid[chaser_x][chaser_y] = 'A'

    def __call__(self):
        #print(self.grid)
        print('\n'.join(map(lambda x: ' '.join(x), self.grid)))