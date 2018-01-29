import numpy as np
from tools.logger import logging_dict, logging_time, logging_number
from tools.parallel import N_process
import time

def RandomWalk(steps , dim=3): 
    '''Using two random choices, determine the dimension and direction of a (steps) steps walk'''
    walks = np.zeros((steps,dim), dtype=int)
    dims = np.random.choice(dim, size=steps)
    ahead = np.random.choice([-1,1], size=steps)
    
    walks[np.arange(steps), dims] = ahead

    pos = np.array(walks).cumsum(axis=0) # cumulative sum of walks in every step
    # returning a 2D array of size (steps, 3)
    return pos

def has_returned(n, dim=3, size=1):
    '''setting numpy seed for different generator, or else they will have the same seed'''
    t = time.time()
    seed = int((t - int(t)) * 1000000)
    np.random.seed( seed)

    ''' Check if a random walk of n steps has returned to its origin point or not'''
    for i in range(size):
        pos = RandomWalk(n, dim=dim) 
        matches = np.all(np.zeros(dim) == pos, axis=1)
        exist_match = np.any( matches )
        yield exist_match

@logging_time
def m_results(m):
    logging_number()
   
    N = 10000
    n = 20000
    dim = 5
   
    func = lambda :sum( has_returned(n, dim, N) ) / N 
    
    '''using python package multiprocessing to fasten the process'''   
    results = N_process(func, m)

    results = np.array(results)
    mean = results.mean()

    std = results.std() 

    logging_dict(locals())
    print(results, mean, std)

m_results(10)

