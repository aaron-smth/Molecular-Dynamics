
def runtime_profile(f, **kwargs):
    import yappi
    import time

    yappi.set_clock_type('cpu')
    yappi.start(builtins=True)

    start = time.time()
    
    f(**kwargs) 

    duration = time.time() - start
    print(f'duration is {duration}')

    stats = yappi.get_func_stats()
    stats.save('callgrind.out', type='callgrind')


def vel_profile(vel):
    import numpy as np
    import matplotlib.pyplot as plt 
    from scipy.stats import maxwell
    speed = np.linalg.norm(vel, axis=1) 
    #hist, bin_edges = np.histogram(speed, density=True, bins=40) 
    #bin_centers = 0.5* (bin_edges[1:] + bin_edges[:-1])

    xs = np.linspace(0.1,max(speed),100) 
    parameters = maxwell.fit(speed, floc=0) 
    print(parameters, speed[:50])

    plt.hist(speed, normed=True, bins=20)
    plt.plot(xs, maxwell.pdf(xs, *parameters),color='red')
    plt.show()
    return  


