import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tools.logger import logging_time, logging_dict, logging_number
import time


def LCG():
    '''LCG Random Number generator'''
    # seed, a, c, m are all prime numbers, to avoid periodicity
    a = 1559
    c = 647
    m = 13229
    
    t = time.time()
    state = int(  (t - int(t)) * 10000  ) % m# This is the seed

      
    logging_dict(locals())

    while True:
        yield state / m 
        state = (a*state+c) % m

#instantiate the LCG
LCG = LCG()

def random_choice(choices, size=1):
    '''Using LCG to randomly choose from a list of objects'''
    def hash(p):
        if type(choices) == int:
            return int(p * choices)
        else:
            return choices[int(p * len(choices))]
    vhash = np.vectorize(hash) # vectorizing hash function to make it operate on a numpy array
    
    random_ns = np.fromiter(LCG, dtype=float, count=size) #from generator LCG draw (size) random numbers
    return vhash(random_ns) 

def RandomWalk(steps, RNG='npRNG', dim=3): 
    '''Using two random choices, determine the dimension and direction of a (steps) steps walk'''
    walks = np.zeros((steps,dim), dtype=int)
    if RNG=='LCG':
        dims = random_choice(dim, size=steps)
        ahead = random_choice([-1,1], size=steps)
    elif RNG=='npRNG':
        dims = np.random.choice(dim, size=steps)
        ahead = np.random.choice([-1,1], size=steps)
    
    walks[np.arange(steps), dims] = ahead

    pos = np.array(walks).cumsum(axis=0) # cumulative sum of walks in every step
    # returning a 2D array of size (steps, 3)
    return pos


'''Below are all testing code'''
def pos_to_sampled_MSD(N, steps, sample_indices, RNG):
    for i in range(N):
        pos = RandomWalk(steps, RNG)[sample_indices,:]
        MSD = np.sum(pos**2, axis=1)
        yield MSD

def samppling(N, walk_steps, sample_steps, RNG):

    sample_size = walk_steps // sample_steps
    sample_indices = np.arange(sample_size) * sample_steps

    print(f'taking the mean date of {N} particles, each walking {walk_steps} steps, sampling every {sample_steps} steps')
    
    
    MSD_mean = np.sum( pos_to_sampled_MSD(N, walk_steps, sample_indices, RNG) ) / N

    return sample_indices, MSD_mean

@logging_time #recording the runtime of this function and recording it in results.log
def linearity_test(RNG, ax):
    logging_number()
    '''1.Generate a 500 particle system, each walking 5000 steps, sampled every 10 steps
       2.Finding the MSD of this system
       3.Linear fit the function (N,MSD) using scipy.optimize.curve_fit
       4.Plotting the fitting line and the data'''
 
    print('Linear fitting of random walk')
    print(f'RNG: {RNG}')
    xs, ys = samppling(500, 5000, 10, RNG)

    def f(x, slope):
        '''Assuming MSD is proportional to N'''
        return slope*x

    # curve_fit: argument:(fitting function, xdata, ydata) return: (best fit parameters, standard deviation)
    parameters, std = curve_fit(f, xs, ys)
    yfit = f(xs, *parameters) # the MSD of the fitted line
 
    print(f'standard error: {std[0][0]}')
    print(f'slope: {parameters[0]}')
    print('expected slope: 1\n')
 
    logging_dict(locals()) #recording the results of this test 

    ax.set(title= RNG, xlabel='steps', ylabel='distance squared')
    ax.plot(xs, ys)
    ax.plot(xs, yfit, color='red', alpha=0.7, label='linear fitting')

# plotting the linear fitting of N walks
def plot_linear(fname):
    '''Plotting the two test together'''
    fig, axes = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(12, 4))
    linearity_test('LCG' ,axes[0] )
    linearity_test('npRNG', axes[1], )
    fig.savefig(fname)

# writting the XYZ file
def write_to_XYZ():
    '''Write a XYZ file using LCG and npRNG'''
    steps = 10000
    LCG_walk = RandomWalk(steps, 'LCG')
    npRNG_walk = RandomWalk(steps, 'npRNG')
    from tools.XYZ_format import write_XYZ
    write_XYZ('LCG.xyz', f'a random walk of {steps} steps using LCG random number generator', LCG_walk)
    write_XYZ('npRNG.xyz', f'a random walk of {steps} steps using numpy random number generator', npRNG_walk)

### Uncomment to plot
if __name__ == "__main__":
    plot_linear('linear.pdf')
    # write_to_XYZ()

