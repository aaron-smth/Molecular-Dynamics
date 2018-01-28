import numpy as np
from lab1 import RandomWalk
from tools.logger import logging_dict
from numba import jit


def has_returned(n, dim=3):
    ''' Check if a random walk of n steps has returned to its origin point or not'''
    while True:
        pos = RandomWalk(n, dim=dim, RNG='npRNG')
        matches = np.all(np.zeros(dim) == pos, axis=1)
        exist_match = np.any( matches )
        yield exist_match


def get_Prob(N, gen):
    '''Conducting a Monte Carlo test of N particles'''
    while True:
        results = np.fromiter( gen, dtype=int, count=N )
        yield sum(results) / N 


def m_results(m):
   
    N = 10000
    n = 20000
    dim = 3
    return_gen =  has_returned(n, dim=dim)
    prob_gen = get_Prob(N, return_gen)
   
    results = np.fromiter(prob_gen, dtype=float, count=m)
    
    mean = results.mean()

    std = results.std() 

    logging_dict(locals())
    return results, mean, std

print(m_results(10))


