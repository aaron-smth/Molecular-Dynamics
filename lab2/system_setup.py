import numpy as np
from numpy import sqrt, log, cos, pi
from tools.XYZ_format import to_xyz
from tools.profiler import vel_profile
'''Units
length: A (Angstrom, 1e-10 m)
Time: ps (picosecons, 1e-12 s)
Energy: ev (electronvolt, 1.602e-19 J)
Mass: u (atomic mass, 931.49 Mev/c^2)

derived quantities:
Light speed: c = 3e6 A/ps
Boltzmann constant: k = 8.617e-5 ev/K
'''


'''All my functions take a dictionary that contains all system information
as an argument'''

def lattice_pos(d):
    N,L = d['N'], d['L']
    dl = 10 # a buffer distance between wall and the particles at the edge
    n_side = int( N ** (1/3) ) +1 # how many particles on a side
    side_coords = np.linspace(0+dl, L-dl, n_side)
    '''meshgrid side_coordinates to get the coordinates'''
    coords = np.vstack(np.meshgrid(side_coords, side_coords, side_coords)).reshape(3,-1).T
    coords = coords[:N] # truncate to N particles
    return coords

def Maxwell_vel(d):
    N,k,T,m = d['N'],d['k'],d['T'],d['m']
    sigma = sqrt(k*T/m)
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
def hard_wall():
    '''A velocity fold and a position fold across the wall '''
    '''state[:, :3] are positions of N particles,
       state[:, 3:] are velocities of N particles'''
    ''' I also shift positions to avoid particles bouncing back and 
    forth outside the box'''
    cond1 = state[:,:3]>L 
    cond2 = state[:,:3]<0
    state[:, :3][cond1] = 2*L - state[:,:3][cond1]
    state[:, 3:][cond1] *= -1
    state[:, :3][cond2] *= -1 
    state[:, 3:][cond2] *= -1
     

def periodic_wall():
    '''A velocity shift and a position shift of distance L towards the center'''
    '''state[:, :3] are positions of N particles,
       state[:, 3:] are velocities of N particles'''
    cond1 = state[:,:3]>L 
    cond2 = state[:,:3]<0
    state[:, :3][cond1] -= L 
    state[:, :3][cond2] += L
  
'''Perform a time evolution of the system'''
def move(): 
    state[:,:3] += state[:,3:] * dt

def evolve(t):
    '''Unit of t: picosecond'''
    steps = t // dt
    with open('frames.xyz', 'wb') as f:
        for i in range(steps):
            to_xyz(f, 'argon atoms', state, name='Ar')   
            hard_wall()
            #periodic_wall()
            move()

'''some statistics of the system'''

'''Initialize an ideal gas system'''
sys_dict = {
    'L' : 1000,
    'N' : 3430,      # number of particles 7**3
    'm' : 39.948,    # argon
    'T' : 200,        # in K
    'k' : 8.617e-5,
    'dt' : 10,
}

state = init_state(sys_dict)

if __name__ == '__main__':
    pass
    #evolve(10000)
    #np.savetxt('vel.csv', np.linalg.norm(state[3:],axis=1) , delimiter='\n')
    vel_profile(state[3:])
