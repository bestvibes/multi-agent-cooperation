import unittest
import numpy as np
import src.rendering

class TestRendering(unittest.TestCase):
    def test_render(self):
        state = ((-1,1),(1,1))
        obstacles = [(-1,0),(0,0)]
        env_size = 1
        renderer = src.rendering.Render2DGrid(obstacles, env_size)

        rendered = renderer(state)

        grid = np.array([['0','.','1'],
                         ['O','O','.'],
                         ['.','.','.']])
        for r1, r2 in zip(grid, rendered):
            for i, j in zip(r1, r2):
                self.assertEqual(i, j)

if __name__ == '__main__':
    unittest.main()