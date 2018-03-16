import numpy as np
import pickle
from numpy import sqrt, log, sin, cos, pi, exp
from tools import logging_dict, logging_time, progress_bar
from tersoff import tersoff_F, tersoff_V

'''Units
length: A (Angstrom, 1e-10 m)
Time: ps (picosecons, 1e-12 s)
Energy: ev (electronvolt, 1.602e-19 J)
Mass: u (atomic mass, 931.49 Mev/c^2)

derived quantities:
Light speed: c = 3e6 A/ps
Boltzmann constant: k = 8.617e-5 ev/K
Carbon_radius: 1.7 A
'''
class MD_sys:

    k = 8.617e-5
    def __init__(self, N, T, steps, atom_info, dt=0.002):
        self.m = atom_info['m']
        self.N = N
        self.T = T
        self.dt = dt
        self.steps = steps # run steps
        self.t = self.dt * self.steps 
        self.L = atom_info['bondLength'] * N**(1/2) * 3 # box size 
        # a buffer distance between wall and the particles at the edge  
        self.dl =  0.05 * self.L 
    
        self.F_memory = np.zeros(3)
        self.V = np.array([])
        self.K = np.array([])
        self.E = np.array([])
        self.count = 0

        self.init_state()

    def lattice_pos(self):
        '''creating initial positions at lattice points, specified by the sys_dict'''
        N,L,dl = self.N, self.L, self.dl
        n_side = np.ceil(N ** (1/3) )   # how many particles on a side
        side_coords = np.linspace(0+dl, L-dl, n_side)
        '''meshgrid side_coordinates to get the coordinates'''
        coords = np.vstack(np.meshgrid(side_coords, side_coords, side_coords)).reshape(3,-1).T
        coords = coords[:N] # truncate to N particles
        return coords
    
    def hexagon_pos(self):
        N, L, dl = self.N, self.L, self.dl

    def Maxwell_vel(self):
        N,k,T,m = self.N, self.k, self.T, self.m 
        sigma = sqrt(k*T/m)
        def Box_Muller(arr): 
            return sqrt(-2*log(arr[0])) * cos(2*pi*arr[1])    

        rands = np.random.random( (N,3,2) ) # (particle_id, v_axis, random u1/u2) 
        vs = np.apply_along_axis(Box_Muller, axis=2, arr=rands) * sigma
        return vs

    def init_state(self, fromF=False):
        xs = self.lattice_pos()
        vs = self.Maxwell_vel()
        '''returning a (N, 6) shaped array representing a state'''
        self.state = np.append( xs, vs, axis=1) 
        '''removing the COM velocity''' 
        self.state[:, 3:] -= np.mean(self.state[:, 3:], axis=0) 
        '''copying the first frame'''
        self.state_li = [self.state.copy()]

    def hard_wall(self):
        '''A velocity fold across the wall '''
        p,v = self.state[:,:3], self.state[:,3:]
        cond1, cond2 = p>self.L , p<0
        v[cond1], v[cond2] = -abs(v[cond1]), abs(v[cond2]) 

    def force(self, new=True):
        '''force calculation'''
        if not new:
            return self.F_memory 

        pos = self.state[:, :3] 
        f = tersoff_F(pos)
        self.F_memory = f

        if self.count%100==0:
            '''recording potential energy every 100 steps'''
            V = tersoff_V(pos) 
            self.V = np.append(self.V , V)
        return f
    
    def HeatBath(self):
        tau = 100 * self.dt
        T_des = self.T
        T_now = self.K[-1] * 2 / (3 * self.k) / self.N
        ratio = np.sqrt( 1 + self.dt / tau * (T_des / T_now -1) )
        self.state[:, 3:] *= ratio
          
    def Verlet(self):
        '''r_n+1 = r_n + v_n * dt + f_n/2m * (dt)^2
           v_n+1 = v_n + (f_n + f_n+1)/2m *dt ''' 
        dt , m = self.dt , self.m 

        f1 = self.force(new=False)
        self.state[:, :3] += self.state[:, 3:] * dt + f1/(2*m) * dt**2
        self.hard_wall()
        f2 = self.force()
        self.state[:, 3:] += (f1+f2) / (2*m) * dt
        #self.HeatBath()

    @logging_time
    def run(self):
        '''main program'''
        for i in range(self.steps):
            progress_bar(i, self.steps) 
            self.K = np.append(self.K ,0.5*self.m*np.sum(self.state[:, 3:]**2))
            self.Verlet()
            if self.count%100==0: # change to 100 in the end
                self.E = np.append(self.E , self.K[-1] + self.V[-1] )
                self.state_li.append(self.state.copy())
            self.count+=1

    def __enter__(self):
        '''before each run, display the system information'''
        logging_dict(self.short_info)

    @property
    def short_info(self):
        return {k:v for k,v in vars(self).items() if type(v) in (int, float)}

    def __str__(self):
        return self.short_info

    def __exit__(self, exc_type, exc_val, exc_tb): 
        '''after every run, dump all info into sys.obj'''
        with open('sys.obj', 'wb') as f:
            pickle.dump(self, f) 

Carbon_info = {
        'm':12.0107, #u
        'bondLength':1.42 #A
        }

def make_sys(steps, new=True):
    if not new:
        with open('sys.obj', 'rb') as f:
            sys = pickle.load(f)
        sys.steps = steps
        sys.t += sys.dt * sys.steps
    else:    
        sys = MD_sys(N=5, T=10, steps=steps, atom_info=Carbon_info)
    return sys

if __name__=='__main__':
    sys = make_sys(steps=20000, new=True)
    with sys:
        sys.run()

