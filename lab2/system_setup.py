import numpy as np
from numpy import sqrt, log, cos, pi
from tools.XYZ_format import to_xyz
'''Units
length: A (Angstrom, 1e-10 m)
Time: ps (picosecons, 1e-12 s)
Energy: ev (electronvolt, 1.602e-19 J)
Mass: u (atomic mass, 931.49 Mev/c^2)

derived quantities:
Light speed: c = 3e6 A/ps
Boltzmann constant: k = 8.617e-5 ev/K
'''

'''Initialize an ideal gas system'''
L = 1000
N = 343 # number of particles 7**3
m = 39.948 # argon
T = 500 # in K
k = 8.617e-5 
dt = 10

def lattice_pos(L, N):

    dl = 10
    n_side = int( N ** (1/3) ) +1
    side_coords = np.linspace(0+dl, L-dl, n_side)
    coords = np.vstack(np.meshgrid(side_coords, side_coords, side_coords)).reshape(3,-1).T
    return coords

def Maxwell_vel(N):

    sigma = sqrt(k*T/m)
    def Box_Muller(arr): return sqrt(-2*log(arr[0])) * cos(2*pi*arr[1]) * sigma

    rands = np.random.random( (N,3,2) ) # (particle_id, v_axis, random u1/u2) 
    vs = np.apply_along_axis(Box_Muller, axis=2, arr=rands)         
    return vs

def init_state():
    xs = lattice_pos(L,N)
    vs = Maxwell_vel(N)
    '''returning a (N, 6) shaped array representing a state'''
    return np.append( xs, vs, axis=1) 

'''Set up a boundary condition'''

def hard_bounce():
    
    def bounce(coord):
        conditions = np.logical_or(coord[:3]<L, coord[:3]>0)
        for i in range(3): 
            if conditions[i]:
                coord[2+i] *= -1
    np.apply_along_axis(bounce, axis=0, arr=state)


def periodic_wall():
    pass 

'''Perform a time evolution of the system'''
def move(): 
    state[:,:3] += state[:, 3:] * dt


state = init_state()
with open('frames.xyz', 'wb') as f:
    for i in range(1000):
        to_xyz(f, 'argon atoms', state, name='Ar')   
        hard_bounce()
        move()
