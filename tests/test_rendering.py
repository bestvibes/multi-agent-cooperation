import unittest
import numpy as np
import src.rendering

def render(state, obstacles, env_size):
    rendering = src.rendering.Render(state[0], state[1], obstacles, env_size)
    return rendering()

class TestRendering(unittest.TestCase):
    def setUp(self):
        self.state = ((-1,1),(1,1))
        self.obstacles = [(-1,0),(0,0)]
        self.env_size = 1

    def test_render(self):
        rendered = render(self.state, self.obstacles, self.env_size)
        grid = np.array([['A','.','T'],
                         ['O','O','.'],
                         ['.','.','.']])
        for r1, r2 in zip(grid, rendered):
            for i, j in zip(r1, r2):
                self.assertEqual(i, j)

if __name__ == '__main__':
    unittest.main()