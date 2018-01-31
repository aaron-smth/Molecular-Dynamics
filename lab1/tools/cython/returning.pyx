import numpy as np
cimport numpy as np

cpdef RandomWalk(int steps): 
    cdef int dim = 3
    '''Using two random choices, determine the dimension and direction of a (steps) steps walk'''
    cdef np.ndarray[long, ndim=2] walks = np.zeros((steps,dim), dtype=long)
    
    cdef np.ndarray[long, ndim=1] dims = np.random.choice(dim, size=steps)
    cdef np.ndarray[long, ndim=1] ahead = np.random.choice([-1,1], size=steps)
    
    walks[np.arange(steps), dims] = ahead

    cdef np.ndarray[long, ndim=2] pos = np.array(walks).cumsum(axis=0) # cumulative sum of walks in every step
    # returning a 2D array of size (steps, 3)
    return pos

cpdef int has_returned(int n):
    cdef int dim = 3
    ''' Check if a random walk of n steps has returned to its origin point or not'''
    cdef np.ndarray[long, ndim=2] pos = RandomWalk(n) 
    cdef int i
    for i in range(n):
        if all(pos[i] == [0,0,0]):
            return True
    return False

cpdef float get_Prob(int N, int n):
    '''Conducting a Monte Carlo test of N particles'''
    cdef int i
    cdef int result = 0
    for i in range(N):
        if has_returned(n):
            result += 1 
    return float(result) / N 

cpdef m_results(int m):
   
    cdef int N = 10000
    cdef int n = 20000
    cdef int dim = 3
    cdef int i
    results = []

    for i in range(m):
        results.append( get_Prob(N, n) ) 
    
    cdef float mean= 0 
    cdef float std = 0
    for i in range(m):
        mean += results[i]
    mean /= m
    for i in range(m):
        std += (results[i]-mean)**2
    std = np.sqrt(std)
        
    print(results, mean, std)


