import numpy as np
from numpy import pi
import matplotlib.pyplot as plt 
import pickle
from tools import to_xyz
from tools import logging_dict
from MD import MD_sys

def moving_ave(arr, cut, n):
    ave = []
    for i in range(cut, arr.size - 1):
        ave.append(arr[i:i+n].mean())
    return ave, cut, n

class MD_sys(MD_sys):
      
    def __enter__(self):
        return self

    @property
    def Ts(self):
        return self.K / self.N / (1.5* self.k) 
    
    def save_data(self, fname):
        '''saving parameters to results.log'''
        logging_dict(self.short_info)
        '''saving xyz frames and energy data'''
        print(f'save data to {fname}')
        with open(fname, 'wb') as f:
            for i,state in enumerate(self.state_li):
                to_xyz(f, f'frame {i}', state, name='Ar')  

    def RDF(self, N_bins=30, ind=-1):
        R = self.L
        pos = self.state_li[ind][:, :3]
        bins = np.linspace(0, R, N_bins)

        all_r = np.array([])
        for i in range(self.N-1):
            r_vecs = pos[i+1:, :] - pos[i, :]
            rs = np.linalg.norm(r_vecs, axis=1) 
            all_r = np.append(all_r, rs)

        hist_ave = np.histogram(all_r, bins=bins)[0]
        hist_ave = hist_ave.astype(float)
        hist_ave *= 2 / self.N * self.L**3 #since we only calculated half the distance

        midbins = (bins[1:] + bins[:-1])/2
        hist_ave /= (4 * pi * midbins)
        return hist_ave, midbins

    @property
    def Correlation(self):
        p0,v0 = self.state_li[0][:, 3:], self.state_li[0][:, :3]
        v_scale = 1 / np.sum(v0**2, axis=1).mean()
        self.Rs = []
        self.Cs = []
        for state in self.state_li:
            p,v = state[:, 3:], state[:, :3]
            R = np.sum((p-p0)**2, axis=1).mean() 
            C = v_scale * np.sum(v*v0 , axis=1).mean()
            self.Rs.append(R)
            self.Cs.append(C) 
        return np.array(self.Rs), np.array(self.Cs)

    def profile_plot(self, fname): 
        t,dt = self.t, self.dt 
        Ts = self.Ts
        
        print('plotting energy data...')    
        fig, axes = plt.subplots(2,3, figsize=(20,10) )
        ax1,ax2,ax3,ax4,ax5,ax6 = axes.reshape(-1)

        for label,energy in zip(['E','V','K'],[self.E,self.V,self.K]):
            ax1.plot(np.linspace(0,t,len(energy)), energy, label=label)
            ax1.axhline(energy[-20:].mean(), ls='--')
        ax1.set(title='Energy evolution')
        ax1.legend()

        ax2.plot(np.linspace(0,t,len(self.Ts)), self.Ts, label='Temperature')
        T_ave,cut,n = moving_ave(Ts,cut=1000,n=200)
        ax2.plot(np.linspace(cut*dt, t-n*dt, len(T_ave)), 
                T_ave, color='red', label='Time average T')
        ax2.axhline(self.T, ls='--', color='orange', label='initial temperature')
        ax2.set(title='Temperature evolution')
        ax2.legend()
        
        indexes = ((len(self.state_li)-1) * np.array([0,0.1,1])).astype(int)
        for ind in indexes: 
            gs, rs = self.RDF(ind=ind)
            ax3.plot(rs[rs<4], gs[rs<4],
                    label=f'time:{round(ind/len(self.state_li), 2)}')
        ax3.set(title='RDF', xlabel='r', ylabel='g(r)')
        ax3.legend(loc='best')

        ax4.plot(np.linspace(0,t,len(self.Correlation[0])), 
                self.Correlation[0], label='R') 
        ax4.set(title='Auto Correlation for displacement', xlabel='t', ylabel='<R(t)^2>')
        ax4.legend()

        ax5.plot(np.linspace(0,t,len(self.Correlation[1])),
                self.Correlation[1], label='C') 
        ax5.set(title='Auto Correlation for velocity', xlabel='t', ylabel='C(t)')
        ax5.legend()

        print(f'saving statistics plots to {fname}')
        fig.savefig(fname)
        
if __name__=='__main__':
    with open('sys.obj', 'rb') as f:
        sys = pickle.load(f)
    
    with sys:
        sys.save_data('frames.xyz')
        sys.profile_plot('statistics.pdf')



