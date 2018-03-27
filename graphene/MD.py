from __future__ import division # To avoid division problems

import numpy as np
import pickle
from numpy import sqrt, log, sin, cos, pi, exp
from tools import logging_dict, logging_time, progress_bar
from tersoff import tersoff_F, tersoff_V
from config import custom_info


'''Units
length: A (Angstrom, 1e-10 m)
Time: ps (picosecons, 1e-12 s)
Energy: ev (electronvolt, 1.602e-19 J)
Mass: u (atomic mass, 931.49 Mev/c^2)
Density: u/A^3 = 1.66054 g/cm^3

derived quantities:
Light speed: c = 3e6 A/ps
Boltzmann constant: k = 8.617e-5 ev/K
Carbon_radius: about 1.7 A
'''

def lattice_pos(N, L):
    dl = 0.05 * L
    '''creating initial positions at lattice points, specified by the sys_dict'''
    n_side = np.ceil(N ** (1/3) )   # how many particles on a side
    side_coords = np.linspace(0+dl, L-dl, n_side)
    '''meshgrid side_coordinates to get the coordinates'''
    coords = np.vstack(np.meshgrid(
        side_coords, side_coords, side_coords)).reshape(3,-1).T
    coords = coords[:N] # truncate to N particles
    return coords

def random_pos(N, L):
    coords = np.random.uniform( 0, L, size=(N, 3))
    return coords 

def hexagon_pos(N, L):
    'For graphene ICs'
    pass

Box_Muller = lambda r1, r2: sqrt(-2*log(r1)) * cos(2*pi*r2)

def Maxwell_vel(N, T, m, k):
    sigma = sqrt(k*T/m)
    rands = np.random.random( (N,3,2) ) # (particle_id, v_axis, random u1/u2) 
    vs = sigma * Box_Muller(rands[:,:,0], rands[:,:,1])
    return vs

class MD_sys:

    k = 8.617e-5
    def __init__(self, custom_info):
        for k, v in custom_info.items():
            setattr(self, k, v)

        self.t = self.dt * self.steps 
        # a buffer distance between wall and the particles at the edge  
    
        self.F_memory = np.zeros((self.N,3))
        self.K = []
        self.r_li = []
        self.v_li = []

        self.init_state()
 
    def init_state(self):
        self.r = random_pos(self.N, self.L)
        self.v = Maxwell_vel(self.N, self.T, self.m, self.k)
        '''removing the COM velocity''' 
        self.v -= self.v.mean(axis=0)

    def hard_wall(self):
        '''A velocity fold across the wall '''
        cond1, cond2 = self.r > self.L , self.r < 0
        self.v[cond1], self.v[cond2] = -abs(self.v[cond1]), abs(self.v[cond2]) 

    def periodic_wall(self):  
        L = self.L
        cond1, cond2 = self.r>L , self.r<0
        self.r[cond1] -= L
        self.r[cond2] += L

    @property
    def force(self):
        f = tersoff_F(self.r)
        self.F_memory = f
        return f
    
    def HeatBath(self):
        tau = 500 * self.dt
        T_des = self.T
        T_now = self.K[-1] * 2 / (3 * self.k) / self.N
        ratio = np.sqrt( 1 + self.dt / tau * (T_des / T_now -1) )
        self.v *= ratio
          
    def Verlet(self):
        '''r_n+1 = r_n + v_n * dt + f_n/2m * (dt)^2
           v_n+1 = v_n + (f_n + f_n+1)/2m *dt ''' 
        dt , m = self.dt , self.m 
        f1 = self.F_memory
        self.r += self.v * dt + f1/(2*m) * dt**2
        self.periodic_wall()
        f2 = self.force
        self.v += (f1+f2) / (2*m) * dt
        if self.HeatBath_on:
            self.HeatBath()

    @logging_time
    def run(self):
        '''main program'''
        for i in range(self.steps):
            progress_bar(i, self.steps) 
            if i%20==0: # change to 100 in the end
                self.K.append(0.5 * self.m * np.sum(self.v**2)) #for heatbath
                self.r_li.append(self.r.copy())
                self.v_li.append(self.v.copy())
            self.Verlet()

    def __enter__(self):
        '''before each run do this'''
        if self.memory_on:
            self.t += self.dt * self.steps 
            if not isinstance(self.K, list):
                self.K = self.K.tolist()
        else: 
            self.t = self.dt * self.steps
            self.K = []
            self.r_li = []
            self.v_li = []
        logging_dict(self.short_info())

    def short_info(self):
        return {k:v for k,v in vars(self).items() if type(v) in (int,float,bool)}

    def __exit__(self, exc_type, exc_val, exc_tb): 
        '''dump all info into sys.obj'''
        with open('sys.obj', 'wb') as f:
            pickle.dump(self, f) 

def make_sys(custom_info):
    fromFile = custom_info['fromFile']
    if fromFile:
        with open(fromFile, 'rb') as f:
            sys = pickle.load(f)
        sys.__dict__.update(custom_info)
    else:    
        sys = MD_sys(custom_info)
    return sys

if __name__ == '__main__':
    sys = make_sys(custom_info)
    with sys:
        sys.run()

