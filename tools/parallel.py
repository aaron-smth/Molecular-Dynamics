import multiprocessing as mp
import numpy as np

def N_process(func, N):
    '''feeding all parameters to a function and do it N times'''
    output = mp.Queue()
    jobs = []
    def f(output):
        result = func() 
        output.put(result)
    for i in range(N):
        p = mp.Process(target = f , args=[output])
        jobs.append(p)
        p.start()
    for i in range(N):
        jobs[i].join()
    results = np.array([output.get() for p in jobs])
    return results

