import numpy as np
from numpy import sqrt, log, cos, pi
from tools.XYZ_format import to_xyz
from tools.logger import logging_dict, logging_time
'''Units
length: A (Angstrom, 1e-10 m)
Time: ps (picosecons, 1e-12 s)
Energy: ev (electronvolt, 1.602e-19 J)
Mass: u (atomic mass, 931.49 Mev/c^2)

derived quantities:
Light speed: c = 3e6 A/ps
Boltzmann constant: k = 8.617e-5 ev/K
Argon_radius: 0.71
'''

'''All my functions take a dictionary that contains all system information
as an argument'''

def lattice_pos(d):
    N,L = d['N'], d['L']
    dl = 0.1*L # a buffer distance between wall and the particles at the edge
    n_side = int( N ** (1/3) ) +1 # how many particles on a side
    side_coords = np.linspace(0+dl, L-dl, n_side)
    '''meshgrid side_coordinates to get the coordinates'''
    coords = np.vstack(np.meshgrid(side_coords, side_coords, side_coords)).reshape(3,-1).T
    coords = coords[:N] # truncate to N particles
    return coords

def Maxwell_vel(d):
    N,k,T,m = d['N'],d['k'],d['T'],d['m']
    sigma = sqrt(k*T/m)
    d['theoretical std'] = sigma
    def Box_Muller(arr): 
        return sqrt(-2*log(arr[0])) * cos(2*pi*arr[1])    

    rands = np.random.random( (N,3,2) ) # (particle_id, v_axis, random u1/u2) 
    vs = np.apply_along_axis(Box_Muller, axis=2, arr=rands) * sigma
    return vs

def init_state(d):
    xs = lattice_pos(d)
    vs = Maxwell_vel(d)
    '''returning a (N, 6) shaped array representing a state'''
    return np.append( xs, vs, axis=1) 

'''Set up a boundary condition'''
def hard_wall(state,d):
    '''A velocity fold and a position fold across the wall '''
    '''state[:, :3] are positions of N particles,
       state[:, 3:] are velocities of N particles'''
    ''' I also shift positions to avoid particles bouncing back and 
    forth outside the box'''
    L = d['L']
    p,v = state[:,:3], state[:,3:]
    cond1 = p>L 
    cond2 = p<0
    v[cond1] = -abs(v[cond1])
    v[cond2] = abs(v[cond2])
     

def periodic_wall(state,d):
    '''A velocity shift and a position shift of distance L towards the center'''
    '''state[:, :3] are positions of N particles,
       state[:, 3:] are velocities of N particles'''
    L = d['L']
    cond1 = state[:,:3]>L 
    cond2 = state[:,:3]<0
    state[:, :3][cond1] -= L 
    state[:, :3][cond2] += L
  
'''Perform a time evolution of the system'''

LJ_memory = np.zeros(3)
def LJ_force(state,  d, new=True, LJ_memory=LJ_memory):
    if not new:
        return LJ_memory
    sig = d['sig']
    eps = d['eps']

    N = d['N']
    L = d['L']

    pos = state[:, :3] 
    fs = np.zeros( (N,3) )

    check_r = False
    for i in range(N):
        r_vecs = pos[i+1:, :] - pos[i, :]
        rs = np.linalg.norm( r_vecs, axis=1 ) 
        
        if len(rs)>=1 and (rs.min() < 0.01*sig or rs.max()>d['L'] ):
            check_r = True
       # eff_ind = rs < 2.25 * sig
       # rs = rs[eff_ind]
       # r_vecs = r_vecs[eff_ind]

        f_mags = 24*eps/rs**2 * (2*(sig/rs)**12 - (sig/rs)**6 )
        f = np.multiply( r_vecs, f_mags.reshape( (-1,1) ) )

        fs[i]    += np.sum( f, axis=0 ) 
        fs[i+1:] -= f
    if check_r:
        d['trouble'] = True
    else:
        d['trouble'] = False
    
    LJ_memory = fs
    return fs

def Verlet(state, d):
    '''r_n+1 = r_n + v_n * dt + f_n/2m * (dt)^2
       v_n+1 = v_n + (f_n + f_n+1)/2m *dt ''' 
    if d['trouble']:
        dt = d['dt_c']
    else:
        dt = d['dt']
    m  = d['m']
    f1 = LJ_force(state, d, new=False)
    f2 = LJ_force(state, d)
    state[:, :3] += state[:, 3:] * dt + f1/(2*m) * dt**2
    state[:, 3:] += (f1+f2) / (2*m) * dt

def evolve(state,d, boundary=hard_wall ): 
    Verlet(state, d)
    state[:, :3] += state[:, 3:] * d['dt']
    '''removing the COM velocity''' 
    state[:, 3:] -= np.mean( state[:, 3:], axis=0 ) 
    boundary(state,d)

@logging_time
def run(d, boundary=hard_wall):
    '''Unit of t: picosecond, t is total runtime'''
    with open('frames.xyz', 'wb') as f:
        for i in range( d['steps'] ):
            if i%10==0:
                to_xyz(f, 'argon atoms', state[:, :3], name='Ar')   
            evolve(state,d) 

'''some statistics of the system'''
def stats(d):
    d['theoretical most probable speed'] = sqrt(2)*d['std']
    d['theoretical mean K.E'] = 1.5*d['k']*d['T']

def vel_profile(vel, d={}):
    import matplotlib.pyplot as plt 
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

'''Initialize an ideal gas system
This dictionary is shared by all functions and modified by all functions
'''
sys_dict = {
    'sig':3.40,
    'eps':1.03e-2,
    'L' : 100,      # Box size
    'N' : 216,      # number of particles 7**3
    'm' : 39.948,    # argon
    'T' : 90,       # in K
    'k' : 8.617e-5,  # Boltzmann Constant
    'steps': 10000,
    't' : 0,
    'dt' : 0.05,       # time of each step
    'dt_c':0.005,
    'trouble':False,
}

state = init_state(sys_dict)
if __name__ == '__main__':
    '''for writting XYZ file'''
    run( sys_dict, boundary=hard_wall)
    '''for plotting velocity profile with matplotlib'''
    #vel_profile(state[:,3:], sys_dict)
    #stats(sys_dict) 
    '''last step, printing all parameters and logging them in results.log'''
    logging_dict(sys_dict) 


