import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import multiprocessing


# seed, a, c, m are all prime numbers, to avoid periodicity
seed = 4211

state = seed # memory of function LCG, keeps updating

a = 1559
c = 313
m = 13229

def LCG():
    '''LCG Random Number generator'''
    global state
    result = (a*state+c) % m
    state = result
    return hashfunc(state)

def hashfunc(n):
    if n/m < 0.5:
        return -1
    else:
        return 1

def npRNG():
    '''Numpy Random Number Generator'''
    if np.random.random() < 0.5:
        return -1
    else:
        return 1

def RandomWalk(steps, RNG):

    walks = []
    for i in range(steps):
        walks.append( [RNG(),RNG(),RNG()] ) # 3D random walk
    walks = np.array(walks)

    pos = walks.cumsum(axis=0) # cumulative sum of walks in every step

    return pos

def N_particle_walk(N, steps, RNG):
    '''Doing random walk for N particles with parallel computing'''
    pool = multiprocessing.Pool(processes=16)

    # feeding RandomWalk the second argument RNG, so that it becomes a single-argument function
    walk = partial(RandomWalk, RNG=RNG) 
    # Using multiple cores, feeding a list of the same variable "steps" to the function "walk"
    particle_positions = pool.map(walk, np.ones(N,dtype=int)*steps)
    
    # Returning an 3D array of size (N, steps, 3), the three indices correspond to (particle index, step index, (x,y,z) coordinate index )
    return np.array(particle_positions)

def pos_to_MSD_mean(N, walk_steps, sample_steps, particle_positions):
    '''1.sample the positions
       2.from positions to mean squared distance, averaged over N particles'''

    sample_size = walk_steps // sample_steps
    sample_indices = np.arange(sample_size) * sample_steps 

    # Returning a sampled version of positions containing only the sampled steps
    sample_positions = particle_positions[:,sample_indices,:] 

    # summing the square of x,y,z coordinates, returning a 2D array (N, MSD)
    MSD_arr =  np.sum(sample_positions**2, axis=2) 
    # summing over MSD of N particles and divide it by number of particles
    MSD_mean = np.sum(MSD_arr,axis=0) / N

    return sample_indices, MSD_mean

def linearity_test(RNG, ax, name=None):
    '''1.Generate a 500 particle system, each walking 5000 steps, sampled every 10 steps
       2.Finding the MSD of this system
       3.Linear fit the function (N,MSD) using scipy.optimize.curve_fit
       4.Plotting the fitting line and the data'''
    N = 500
    walk_steps = 5000
    sample_steps = 10
    print('Linear fitting of random walk')
    print('RNG: {}'.format(name))
    print('taking the mean date of {} particles, each walking {} steps, sampling every {} steps'.format(N, walk_steps, sample_steps))
    particle_positions = N_particle_walk(N, walk_steps, RNG)
    xs, MSDs_mean = pos_to_MSD_mean(N, walk_steps, sample_steps, particle_positions)

    def f(x, slope):
        '''Assuming MSD is proportional to N'''
        return slope*x

    # curve_fit: argument:(fitting function, xdata, ydata) return: (best fit parameters, standard deviation)
    parameters, std = curve_fit(f, xs, MSDs_mean)
    linear = f(xs,*parameters) # the MSD of the fitted line
 
    print('standard error: {}'.format(*std))
    print('slope: {}'.format(*parameters))
    print('expected slope: 3\n')

    ax.set(title= name, xlabel='steps', ylabel='distance squared')
    ax.plot(xs, MSDs_mean)
    ax.plot(xs, linear, color='red', alpha=0.7, label='linear fitting')

# plotting the linear fitting of N walks
def plot_linear():
    '''Plotting the two test together'''
    fig, axes = plt.subplots(nrows=1,ncols=2, sharey=True, figsize=(12, 4))
    linearity_test(LCG, axes[0], 'LCG')
    linearity_test(npRNG, axes[1], 'numpy RNG')
    plt.show(fig)

### Uncomment to plot
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

