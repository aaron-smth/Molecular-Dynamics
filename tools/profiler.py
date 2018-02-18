
import numpy as np
import matplotlib.pyplot as plt 

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


def vel_profile(vel, d={}):
    from scipy.stats import maxwell
    speed = np.linalg.norm(vel, axis=1) 
    #hist, bin_edges = np.histogram(speed, density=True, bins=40) 
    #bin_centers = 0.5* (bin_edges[1:] + bin_edges[:-1])

    xs = np.linspace(0,max(speed),100) 
    mean, std = maxwell.fit(speed, floc=0) 
    fit = maxwell.pdf(xs, mean, std)    
    
    d['std'] = std
    d['K.E'] = 0.5 * d['m'] * (sum(speed**2) / d['N'])
    d['most probable Speed'] = xs[ np.argmax(fit) ]

    plt.hist(speed, normed=True, bins=30)
    plt.plot(xs, fit, color='red')
    plt.show() 

def energy_profile(d):
    E, V, K = d['E'], d['V'], d['K']
    T, Ts = d['T'], d['Ts']
    t = d['t']

    fig, axes = plt.subplots(1,2, figsize=(20,6) )
    ax1, ax2 = axes

    ax1.plot(E, label='Total Energy')
    ax1.plot(V, label='Potential Energy', color='blue')
    ax1.axhline(V[-20:].mean(), ls='--', color='blue')
    ax1.plot(K, label='Kinetic Energy', color='red')
    ax1.axhline(K[-20:].mean(), ls='--', color='red')
    ax1.set(title='Energy evolution')

    ax1.legend()

    ax2.plot(Ts, label='Temperature')
    ax2.axhline(Ts[-20:].mean(), ls='--')
    ax2.axhline(T, ls='--', color='orange', label='initial temperature')
    ax2.set(title='Temperature evolution')

    ax2.legend()

    fname = 'energy.pdf'
    print(f'saving energy profile to {fname}')
    fig.savefig(fname)

