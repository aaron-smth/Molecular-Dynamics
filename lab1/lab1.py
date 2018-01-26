import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import logging, sys, time

LOG_FORMAT = '%(asctime)s - %(message)s'
logging.basicConfig(filename='linearity_test_results.log', level=logging.DEBUG, format=LOG_FORMAT)
logger = logging.getLogger()

def LCG():
    '''LCG Random Number generator'''
    # seed, a, c, m are all prime numbers, to avoid periodicity
    state = 4211 # This is the seed

    a = 1559
    c = 313
    m = 13229
    while True:
        if state / m < 0.5:
            yield -1
        else:
            yield 1
        state = (a*state+c) % m

def npRNG():
    '''Numpy Random Number Generator'''
    while True:
        if np.random.random() < 0.5:
            yield -1
        else:
            yield 1
LCG = LCG()
npRNG = npRNG()


def RandomWalk(steps, RNG): 
    walks = list(next(RNG) for i in range(steps*3))
    walks = np.array(walks).reshape(-1,3)
    pos = np.array(walks).cumsum(axis=0) # cumulative sum of walks in every step
    return pos

def sampled_MSD(N, steps, sample_indices, RNG):
    for i in range(N):
        pos = RandomWalk(steps, RNG)[sample_indices,:]
        MSD = np.sum(pos**2, axis=1)
        yield MSD

def linearity_test(RNG, ax, name=None):
    start = time.time()
    '''1.Generate a 500 particle system, each walking 5000 steps, sampled every 10 steps
       2.Finding the MSD of this system
       3.Linear fit the function (N,MSD) using scipy.optimize.curve_fit
       4.Plotting the fitting line and the data'''
    N = 500
    walk_steps = 5000
    sample_steps = 10
    sample_size = walk_steps // sample_steps
    sample_indices = np.arange(sample_size) * sample_steps

    print('Linear fitting of random walk')
    print('RNG: {}'.format(name))
    print('taking the mean date of {} particles, each walking {} steps, sampling every {} steps'.format(N, walk_steps, sample_steps))
    
    MSD_mean = np.sum(sampled_MSD(N, walk_steps, sample_indices, RNG)) / N
    xs = sample_indices

    def f(x, slope):
        '''Assuming MSD is proportional to N'''
        return slope*x

    # curve_fit: argument:(fitting function, xdata, ydata) return: (best fit parameters, standard deviation)
    parameters, std = curve_fit(f, xs, MSD_mean)
    linear = f(xs,*parameters) # the MSD of the fitted line
 
    print('standard error: {}'.format(*std))
    print('slope: {}'.format(*parameters))
    print('expected slope: 3\n')
    
    def logging_info(d):
        '''Logging the parameters and outputs of this function'''
        li = [(k,v) for k,v in d.items() if sys.getsizeof(v)<1000 and not callable(v)] # logging varoables that are not too big and not functions
        message = '{}: {}' 
        logger.info(', '.join([message.format(*tup) for tup in li]))
        duration = time.time() - start
        logging.info('duration : {}'.format(duration)) # logging runtime
    logging_info(locals()) # logging local variables
    

    ax.set(title= name, xlabel='steps', ylabel='distance squared')
    ax.plot(xs, MSD_mean)
    ax.plot(xs, linear, color='red', alpha=0.7, label='linear fitting')

# plotting the linear fitting of N walks
def plot_linear():
    '''Plotting the two test together'''
    fig, axes = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(12, 4))
    linearity_test(LCG, axes[0], 'LCG')
    linearity_test(npRNG, axes[1], 'numpy RNG')
    plt.show(fig)
    print('The seed is predetermined for function LCG, so the slope of LCG is the same everytime, but npRNG changes everytime you run.\
            the slope it generally approaches 3, but sometimes go beyond 3')

### Uncomment to plot
if __name__ == "__main__":
    plot_linear()

# writting the XYZ file
def write_to_XYZ():
    steps = 5000
    LCG_walk = RandomWalk(steps, LCG)
    npRNG_walk = RandomWalk(steps, npRNG)
    from XYZ_format import write_XYZ
    write_XYZ('LCG.xyz', 'a random walk of 5000 steps using LCG random number generator', LCG_walk)
    write_XYZ('npRNG.xyz', 'a random walk of 5000 steps using numpy random number generator', npRNG_walk)

# write_to_XYZ()

