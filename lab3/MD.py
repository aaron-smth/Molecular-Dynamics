import numpy as np
import sys
from numpy import sqrt, log, cos, pi
from tools.XYZ_format import to_xyz, from_xyz
from tools.logger import logging_dict, logging_time, progress_bar
from tools.profiler import energy_profile

'''Units
length: A (Angstrom, 1e-10 m)
Time: ps (picosecons, 1e-12 s)
Energy: ev (electronvolt, 1.602e-19 J)
Mass: u (atomic mass, 931.49 Mev/c^2)

derived quantities:
Light speed: c = 3e6 A/ps
Boltzmann constant: k = 8.617e-5 ev/K
Argon_radius: 0.71 A
'''

'''All my functions take a dictionary that contains all system information
as an argument'''
   
def lattice_pos(d):
    '''creating initial positions at lattice points, specified by the sys_dict'''
    N,L = d['N'], d['L']
    dl = 0.05*L # a buffer distance between wall and the particles at the edge  
    n_side = np.ceil(N ** (1/3) )   # how many particles on a side
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

def init_state(d,fromF=False):
    if fromF:
        '''loading the last frame of an xyz file and take it as the initial state'''
        print(f'reading from file {fromF}')
        return from_xyz(fromF)
    xs = lattice_pos(d)
    vs = Maxwell_vel(d)
    '''returning a (N, 6) shaped array representing a state'''
    return np.append( xs, vs, axis=1) 

'''Set up a boundary condition'''
def hard_wall(state,d):
    '''A velocity fold across the wall '''
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

def LJ_force(state,  d, new=True ):
    '''Lennard-Jones Potential, force calculation'''
    if not new:
        return d['LJ_memory']
    sig = d['sig']
    eps = d['eps']
    N = d['N']
    L = d['L']

    pos = state[:, :3] 
    fs = np.zeros( (N,3) )

    V = 0 # potentia energy
    for i in range(N-1):
        r_vecs = pos[i+1:, :] - pos[i, :]
        rs = np.linalg.norm( r_vecs, axis=1 ) 
       
        if d['count']%100==0:
            V += np.sum(4*eps*( (sig/rs)**12 - (sig/rs)**6 ))
        f_mags = -24*eps/rs**2 * (2*(sig/rs)**12 - (sig/rs)**6 )
        f = np.multiply( r_vecs, f_mags.reshape( (-1,1) ) )

        fs[i]    += np.sum( f, axis=0 ) 
        fs[i+1:] -= f
   
    d['LJ_memory'] = fs
    if d['count']%100==0:
        '''recording potential energy every 100 steps'''
        d['V'] = np.append(d['V'], V)
    return fs

def Verlet(state, d):
    '''r_n+1 = r_n + v_n * dt + f_n/2m * (dt)^2
       v_n+1 = v_n + (f_n + f_n+1)/2m *dt ''' 
    dt = d['dt']
    m  = d['m']
    boundary = d['boundary']

    f1 = LJ_force(state, d, new=False)
    state[:, :3] += state[:, 3:] * dt + f1/(2*m) * dt**2
    boundary(state,d)
    f2 = LJ_force(state, d)
    state[:, 3:] += (f1+f2) / (2*m) * dt


@logging_time
def run(d):
    '''Unit of t: picosecond, t is total runtime'''
    d['t'] = d['dt'] * d['steps']
    steps = d['steps'] 
    '''removing the COM velocity''' 
    state[:, 3:] -= np.mean( state[:, 3:], axis=0 ) 

    '''main program'''
    for i in range( steps ):
        progress_bar(i, steps ) 
        Verlet(state,d) 
        if d['count']%100==0: # change to 100 in the end
            d['K'] = np.append(d['K'] ,0.5* d['m'] * np.sum(state[:, 3:]**2) )
            d['E'] = np.append(d['E'] , d['K'][-1] + d['V'][-1] )
            '''end program if system energy blows up'''
            if len(d['E'])>5 and abs((d['E'][-1]-d['E'][0])/d['E'][0]) > 10:
                print('system blows up.')
                save_data(state_li, 'frames.xyz', d)
                logging_dict(d) 
                sys.exit() 
            state_li.append(state.copy())
        d['count']+=1

 
def save_data(state_li, fname, d): 
    '''saving xyz frames and energy data'''
    print(f'save data to {fname}')
    with open(fname, 'wb') as f:
        for i,state in enumerate(state_li):
            to_xyz(f, f'frame {i}', state, name='Ar')  
    print('plotting energy data...')    
    d['Ts'] = d['K'] / d['N'] / (1.5 *d['k'] )
    energy_profile(d)     

'''
This dictionary is shared by all functions and modified by all functions
'''
sys_dict = {
    'sig':3.40,      # sigma in LJ potential (regarded as a general length scale)
    'eps':1.03e-2,   # epsilon in LJ potential
    'k' : 8.617e-5,  # Boltzmann Constant
    'm' : 39.948,    # atomic mass of argon
    'boundary':hard_wall,
    'LJ_memory':np.zeros(3),
    'V':np.array([]),
    'K':np.array([]),
    'E':np.array([]),
    'count':0,
    'N' : 216,      # number of particles
    'T' : 5,       # in K
    'steps': 20000,
    'dt' : 0.02, # time elapse of each step
}
'''adaptive box size, using sigma as the length scale'''
sys_dict['L'] =  sys_dict['sig'] * sys_dict['N']**(1/3) * 1.05

print('system information:\n')
for k,v in sys_dict.items():
    print(f'{k}: {v}')


#state = init_state(sys_dict, fromF='frames.xyz')
state = init_state(sys_dict)
state_li = [] # contains all the frames the xyz file need
if __name__ == '__main__':
    run( sys_dict ) 
    '''printing all parameters and logging them in results.log'''
    logging_dict(sys_dict) 
    '''saving data in xyz file and making energy plot'''
    save_data(state_li, 'frames.xyz', sys_dict)

