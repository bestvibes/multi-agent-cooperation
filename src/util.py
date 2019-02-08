import math
import random
from itertools import chain
from collections.abc import Iterable

import torch

def get_coordinates_l2_dist(c1: tuple, c2: tuple):
    if len(c1) != len(c2):
        raise ValueError("Coordinate dimension mismatch!")

    num_dims = len(c1)
    dim_dists = map(lambda i: c1[i] - c2[i], range(num_dims))
    l2_dist = math.sqrt(sum(map(lambda d: d**2, dim_dists)))

    return l2_dist

def list_batch_random_sample(l: list, batch_size: int):
	return random.sample(l, batch_size)

def flatten_tuple(t: tuple):
    if (isinstance(t, Iterable)):
        # NOTE: this assumes only up to one level of nesting: eg (1,2,3,4) or ((1,2), (3,4))
        return list(chain(*t))
    else:
        return t
