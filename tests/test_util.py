import unittest
import math

import src.util

class TestCoordinatesL2Dist(unittest.TestCase):
    def test_dim_mismatch(self):
        self.assertRaises(ValueError,
                            src.util.get_coordinates_l2_dist,
                            (1,2,3),
                            (4,5))

    def test_linear_l2_dist(self):
        l2_dist = src.util.get_coordinates_l2_dist((0,1), (0,5))
        self.assertEqual(l2_dist, 4)

    def test_diagonal_l2_dist(self):
        l2_dist = src.util.get_coordinates_l2_dist((0,1), (2,5))
        self.assertEqual(l2_dist, math.sqrt(20))
