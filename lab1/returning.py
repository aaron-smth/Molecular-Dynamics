import numpy as np
from tools.logger import logging_dict, logging_time

def RandomWalk(steps , dim=3): 
    '''Using two random choices, determine the dimension and direction of a (steps) steps walk'''
    walks = np.zeros((steps,dim), dtype=int)
    dims = np.random.choice(dim, size=steps)
    ahead = np.random.choice([-1,1], size=steps)
    
    walks[np.arange(steps), dims] = ahead

    pos = np.array(walks).cumsum(axis=0) # cumulative sum of walks in every step
    # returning a 2D array of size (steps, 3)
    return pos


def has_returned(n, dim=3):
    ''' Check if a random walk of n steps has returned to its origin point or not'''
    while True:
        pos = RandomWalk(n, dim=dim) 
        matches = np.all(np.zeros(dim) == pos, axis=1)
        exist_match = np.any( matches )
        yield exist_match


def get_Prob(N, gen):
    '''Conducting a Monte Carlo test of N particles'''
    while True:
        results = np.fromiter( gen, dtype=int, count=N )
        yield sum(results) / N 

@logging_time
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
    print(results, mean, std)

m_results(10)

