import numpy as np
from numpy import pi
import matplotlib.pyplot as plt 
import pickle
from tools import to_xyz
from tools import logging_dict
from MD import MD_sys

with open('sys.obj', 'rb') as f:
    sys = pickle.load(f)

def save_data(self,fname):
    '''saving parameters to results.log'''
    logging_dict(vars(self))
    '''saving xyz frames and energy data'''
    print(f'save data to {fname}')
    with open(fname, 'wb') as f:
        for i,state in enumerate(self.state_li):
            to_xyz(f, f'frame {i}', state, name='Ar')  

def moving_ave(arr, cut, n):
    ave = []
    for i in range(cut, arr.size - 1 ):
        ave.append( arr[i:i+n].mean() )
    return ave, cut, n

def make_RDF(self, N_bins):
    R = self.L
    pos = self.state[:, :3]
    bins = np.linspace(0, R, N_bins)

    all_r = np.array([])
    for i in range(self.N-1):
        r_vecs = pos[i+1:, :] - pos[i, :]
        rs = np.linalg.norm( r_vecs, axis=1 ) 
        all_r = np.append( all_r, rs )

    hist_ave = np.histogram(all_r, bins=bins)[0]
    hist_ave = hist_ave.astype(float)
    hist_ave *= 2 / self.N * self.L**3 #since we only calculated half the distance

    midbins = (bins[1:] + bins[:-1])/2
    hist_ave /= ( 4 * pi * midbins )
    self.RDF = (hist_ave, midbins)

def autocorrelation(self):
    p0,v0 = self.state_li[0][:, 3:], self.state_li[0][:, :3]
    v_scale = np.sum(1/v0**2, axis=1).mean()
    self.Rs = []
    self.Cs = []
    for state in self.state_li[1:]:
        p,v = state[:, 3:], state[:, :3]
        R = np.sum( (p-p0)**2, axis=1).mean() 
        C = v_scale * np.sum( v * v0 , axis=1).mean()
        self.Rs.append(R)
        self.Cs.append(C) 
    self.Rs = np.array(self.Rs)
    self.Cs = np.array(self.Cs)
    print(self.Rs, self.Cs)

def profile_plot(self): 
    self.Ts = self.K / self.N / (1.5* self.k) 
    T, Ts = self.T, self.Ts 
    gs, rs = self.RDF 
    t,dt = self.t, self.dt 
    
    print('plotting energy data...')    
    fig, axes = plt.subplots(2,3, figsize=(18,10) )
    ax1,ax2,ax3,ax4,ax5,ax6 = axes.reshape(-1)

    for label,energy in zip(['E','V','K'],[self.E,self.V,self.K]):
        ax1.plot(np.linspace(0,t,len(energy)), energy, label=label)
        ax1.axhline(energy[-20:].mean(), ls='--')
    ax1.set(title='Energy evolution')
    ax1.legend()

    ax2.plot(np.linspace(0,t,len(Ts)), Ts, label='Temperature')

    T_ave,cut,n = moving_ave(Ts,cut=10000,n=2000)
    ax2.plot(np.linspace(cut*dt, t-n*dt, len(T_ave)), \
            T_ave, color='red', label='Time average T')
    ax2.axhline(T, ls='--', color='orange', label='initial temperature')
    ax2.set(title='Temperature evolution')
    ax2.legend()

    ax3.plot(rs, gs)
    ax3.set(title='RDF', xlabel='r', ylabel='g(r)')

    ax4.plot(np.linspace(0,t,len(self.Rs)), self.Rs, label='R') 
    ax4.set(title='Auto Correlation for displacement', xlabel='t')
    ax4.legend()

    ax5.plot(np.linspace(0,t,len(self.Cs)), self.Cs, label='C') 
    ax5.set(title='Auto Correlation for velocity', xlabel='t')
    ax5.legend()

    fname = 'energy.pdf'
    print(f'saving energy profile to {fname}')
    fig.savefig(fname)


if __name__=='__main__':
    make_RDF(sys, 30)
    autocorrelation(sys)
    save_data(sys, 'frames.xyz')

    profile_plot(sys)



