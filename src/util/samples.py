import math
import random
from itertools import combinations
import bisect
import numpy as np

from project_config.project_config import SEED

random.seed(SEED)

def random_combinations(iterable, k, size):
    "Sample n=size combinations of k elements from iterable without replacement."
    pool = tuple(iterable)
    n = len(pool)
    m = (math.comb(n,k))
    codes = random.sample(range(m), size)
    l = list()
    # Given j=code, get the j-th combination
    for code in codes:
        indices = np.array([], dtype=int)
        j = code
        for i in range(k):
            q, r = divmod(j, n-i)
            index = bisect.bisect_right(indices, r)
            r = r + index
            indices = np.insert(indices, index, r)
            j = q
        l.append(pool[i] for i in list(indices)) 
    return l