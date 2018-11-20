import numpy as np
import time

class Render():
    def __init__(self, chaser_pos, target_pos, env_size):
        self.env_size = env_size
        self.target_y = target_pos[1] + self.env_size
        self.target_x = self.env_size - target_pos[0]
        
        self.grid = np.ones((2 * self.env_size+1, 2 * self.env_size + 1))
        self.grid[self.target_x][self.target_y] = 2

        chaser_y = chaser_pos[1] + self.env_size
        chaser_x = self.env_size - chaser_pos[0]

        self.grid[chaser_x][chaser_y] = 0

    def __call__(self):
        print(self.grid)

# def main():
#     target_pos = (3, 3)
#     chaser_pos = (0, 0)
#     env_size = 5
#     max_iter = 5
#     for i in range(max_iter):
#         new_pos = (chaser_pos[0] + i, chaser_pos[1] + i)
#         # new_pos = (1, 1)
#         rendering = Render(new_pos, target_pos, env_size)
#         time.sleep(0.5)
#         rendering()
#         # print("iter:", i)#, end='\r')
#         # print(rendering())#, end="\r")

# if __name__ == '__main__':
#     main()
