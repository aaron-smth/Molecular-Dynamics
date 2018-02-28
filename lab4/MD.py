import numpy as np
from numpy import sqrt, log, cos, pi
import pickle
from tools import logging_dict, logging_time, progress_bar

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
class MD_sys:

    k = 8.617e-5
    def __init__(self, N, T, steps, atom_info, dt=0.1):
        self.sig = atom_info['sig']
        self.eps = atom_info['eps']
        self.m   = atom_info['m']
        self.N = N
        self.T = T
        self.dt = dt
        self.steps = steps # run steps
        self.L =  self.sig * self.N**(1/3) * 1.5 # box size determined by N
        # a buffer distance between wall and the particles at the edge  
        self.dl =  0.05 * self.L 

        self.LJ_memory = np.zeros(3)
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

    def Maxwell_vel(self):
        N,k,T,m = self.N, self.k, self.T, self.m 
        sigma = sqrt(k*T/m)
        def Box_Muller(arr): 
            return sqrt(-2*log(arr[0])) * cos(2*pi*arr[1])    

        rands = np.random.random( (N,3,2) ) # (particle_id, v_axis, random u1/u2) 
        vs = np.apply_along_axis(Box_Muller, axis=2, arr=rands) * sigma
        return vs

    def init_state(self, fromF=False):
        if fromF:
            '''loading the last frame of an xyz file and take it as the initial state'''
            print(f'reading from file {fromF}')
            self.state = from_xyz(fromF)
        else:
            xs = self.lattice_pos()
            vs = self.Maxwell_vel()
            '''returning a (N, 6) shaped array representing a state'''
            self.state = np.append( xs, vs, axis=1) 

        print('system info:')
        logging_dict(vars(self))

        self.state_li = [self.state.copy()]

    def hard_wall(self):
        '''A velocity fold across the wall '''
        p,v = self.state[:,:3], self.state[:,3:]
        cond1, cond2 = p>self.L , p<0
        v[cond1], v[cond2] = -abs(v[cond1]), abs(v[cond2]) 

    def LJ_force(self, new=True):
        '''Lennard-Jones Potential, force calculation'''
        if not new:
            return self.LJ_memory 
        sig,eps,N,L = self.sig,self.eps,self.N,self.L 

        pos = self.state[:, :3] 
        fs = np.zeros( (N,3) )

        V = 0 # potentia energy
        for i in range(N-1):
            r_vecs = pos[i+1:, :] - pos[i, :]
            rs = np.linalg.norm( r_vecs, axis=1 ) 
           
            if self.count%100==0:
                V += np.sum(4*eps*( (sig/rs)**12 - (sig/rs)**6 ))
            f_mags = -24*eps/rs**2 * (2*(sig/rs)**12 - (sig/rs)**6 )
            f = np.multiply( r_vecs, f_mags.reshape( (-1,1) ) )

            fs[i]    += np.sum( f, axis=0 ) 
            fs[i+1:] -= f
       
        self.LJ_memory = fs
        if self.count%100==0:
            '''recording potential energy every 100 steps'''
            self.V = np.append(self.V , V)
        return fs
    
    def HeatBath(self):
        tau = 100 * self.dt
        T_des = self.T
        T_now = self.K[-1] * 2 / (3 * self.k) / self.N
        ratio = np.sqrt( 1 + self.dt / tau * (T_des / T_now -1) )
        self.state[:, 3:] *= ratio
         
    
    def Verlet(self):
        '''r_n+1 = r_n + v_n * dt + f_n/2m * (dt)^2
           v_n+1 = v_n + (f_n + f_n+1)/2m *dt ''' 
        dt,m = self.dt,self.m 

        f1 = self.LJ_force(new=False)
        self.state[:, :3] += self.state[:, 3:] * dt + f1/(2*m) * dt**2
        self.hard_wall()
        f2 = self.LJ_force()
        self.state[:, 3:] += (f1+f2) / (2*m) * dt
        self.HeatBath()


    @logging_time
    def run(self):
        '''Unit of t: picosecond, t is total runtime'''
        self.t = self.dt * self.steps 
        '''removing the COM velocity''' 
        self.state[:, 3:] -= np.mean( self.state[:, 3:], axis=0 ) 

        '''main program'''
        for i in range( self.steps ):
            progress_bar(i, self.steps ) 
            self.K = np.append(self.K ,0.5* self.m * np.sum(self.state[:, 3:]**2) )
            self.Verlet()
            if self.count%100==0: # change to 100 in the end
                self.E = np.append(self.E , self.K[-1] + self.V[-1] )
                self.state_li.append(self.state.copy())
            self.count+=1
            

Argon_info = {
    'sig':3.40,
    'eps':1.03e-2,
    'm':39.948
    }

def from_obj(steps, new=True):
    if not new:
        with open('sys.obj', 'rb') as f:
            sys = pickle.load(f)
        sys.steps = steps
    else:    
        sys = MD_sys(N=216, T=100, steps=steps, atom_info=Argon_info)
    return sys

if __name__=='__main__':
    sys = from_obj(20000, new=False)
    sys.run()
    with open('sys.obj', 'wb') as f:
        pickle.dump(sys,f) 

