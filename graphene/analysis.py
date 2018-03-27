import numpy as np
from numpy import pi
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt 
import pickle
from tools import to_xyz, logging_dict
from MD import MD_sys
from tersoff import tersoff_V
from itertools import groupby

def percent_ind(li, percents):
    inds = (int( (len(li)-1) * i) for i in percents)
    for ind in inds:
        yield li[ind]

def moving_ave(arr, cut, n):
    ave = []
    for i in range(cut, arr.size - 1):
        ave.append(arr[i:i+n].mean())
    return ave, cut, n

def temperature(K, N, k):
    return K / N / (1.5 * k) 

def hist(xs, boundary=None, bins='sqrt'):
    fs, bins = np.histogram(xs, range=boundary, bins=bins)
    midpoints = (bins[1:] + bins[:-1])/2
    fs = fs.astype(float)
    return midpoints, fs

def plot_peaks(xs, ys, ax):
    width = 15
    peaks = find_peaks_cwt(ys, np.arange(1,width)) 
    for xy in zip(xs[peaks], ys[peaks]):
        ax.annotate('(%0.2f, %0.2f)' % xy, xy=xy, textcoords='data')

def RDF(pos, boundary=(0,4)):
    N = len(pos)
    pair_rs = pdist(pos) #pairwise distance
    R_max = pair_rs.max()
    rs, gs = hist(pair_rs, boundary)
    # 1.from pairwise to all 2.from volume density to linear density
    gs *= 2 / N * R_max**3 / (4 * pi * rs) 
    return rs, gs

def Correlation(r_li, v_li):
    p0,v0 = r_li[0], v_li[0]
    v_scale = 1 / np.sum(v0**2, axis=1).mean()
    Rs = []
    Cs = []
    for p,v in zip(r_li, v_li):
        R = np.sum((p-p0)**2, axis=1).mean() 
        C = v_scale * np.sum(v*v0 , axis=1).mean()
        Rs.append(R)
        Cs.append(C) 
    return np.array(Rs), np.array(Cs)

def Neighbors(pos):
    from tersoff import get_relative, R, D
    rij, dij = get_relative(pos)
    neighbors = np.sum(dij < R + D, axis=1)
    return neighbors

def angle_gen(pos):
    from tersoff import get_relative, theta_func, R, D
    rij, dij = get_relative(pos)
    cut_mask = dij < R+D
    cut_ind = np.where(cut_mask)
    cut_groups = np.split(np.arange(len(cut_ind[0])),
        np.where( np.diff(cut_ind[0]) )[0]+1)
    cut_groups.sort(key=len) #for later
    riN = rij[cut_ind]
    diN = dij[cut_ind]
    for N, inds_iter in groupby(cut_groups, key=len):
        inds = np.vstack(list(inds_iter))
        mask = np.triu(np.ones((N,N), dtype=bool), k=1)
        if N > 1:
            for mat in theta_func(riN[inds], diN[inds]):
                yield mat[mask]

def ADF(pos):
    from functools import reduce
    gen = angle_gen(pos)
    Ctheta = reduce(lambda x,y:np.append(x,y), gen)
    theta = np.nan_to_num(np.arccos(Ctheta)) * 180 / np.pi
    return theta

def Virial_K(pos):
    from tersoff import tersoff_F
    force = tersoff_F(pos)
    V = np.einsum('id, id' ,pos, force)
    return V

def Pressure(pos):
    pass

class MD_sys(MD_sys):
      
    @property
    def volume(self):
        return self.L**3

    def __enter__(self):
        self.K = np.array(self.K)
        self.V = np.fromiter((tersoff_V(r) for r in self.r_li), 
                dtype=float)
        self.E = self.V + self.K
        self.Ts = temperature(self.K, self.N, self.k)

        return self    
 
    def save_data(self, fname):
        '''saving parameters to results.log'''
        logging_dict(self.short_info())
        '''saving xyz frames and energy data'''
        print(f'save data to {fname}')
        with open(fname, 'wb') as f:
            for i,r in enumerate(self.r_li):
                to_xyz(f, f'frame {i}', r, name='C')  

    def profile_plot(self, fname): 
        t,dt,N = self.t, self.dt, self.N
        
        print('plotting energy data...')    
        fig, axes = plt.subplots(2,3, figsize=(20,10) )
        ax1,ax2,ax3,ax4,ax5,ax6 = axes.reshape(-1)

        for label,energy in zip(['E','V','K'],[self.E,self.V,self.K]):
            ax1.plot(np.linspace(0,t,len(energy)), energy / N , label=label)
        ax1.set(title='Energy evolution', xlabel='t/ps', ylabel='Energy ev/atom')
        ax1.legend()

        ax2.plot(np.linspace(0,t,len(self.Ts)), self.Ts, label='Temperature')
        T_ave,cut,n = moving_ave(self.Ts,cut=1000,n=200)
        ax2.plot(np.linspace(cut*dt, t-n*dt, len(T_ave)), 
                T_ave, color='red', label='Time average T')
        ax2.axhline(self.T, ls='--', color='orange', label='initial temperature')
        ax2.set(title='Temperature evolution')
        ax2.legend()
        
        percents = [0, 0.2, 0.5, 1]
        for i,pos in enumerate(percent_ind(self.r_li, percents)):
            rs, gs = RDF(pos)
            ax3.plot(rs, gs, label=f'time:{round(self.t * percents[i],2)}')
        plot_peaks(rs, gs, ax3)
        ax3.set(title='RDF', xlabel='r', ylabel='g(r)')
        ax3.legend(loc='best')

        for i,r in enumerate(percent_ind(self.r_li, percents)):
            thetas = ADF(r)
            xs, ys = hist(thetas)
            ax4.plot(xs, ys,label=f'time:{round(self.t * percents[i],2)}')
            plot_peaks(xs, ys, ax4)
            ax4.set(xlabel='angles', ylabel='frequency')
        ax4.legend()

        neighbors = Neighbors(self.r)
        xs, ys = hist(neighbors, boundary=(0,6))
        ax5.plot(xs, ys)
        ax5.annotate(f'mean:{neighbors.mean()}', xy=(0.8,0.8),
                textcoords='axes fraction')
        ax5.set(xlabel='number of neighbors', ylabel='frequency')

        print(f'saving statistics plots to {fname}')
        fig.savefig(fname)

if __name__ == '__main__':
    with open('sys.obj', 'rb') as f:
        sys = pickle.load(f) 
    with sys:
        sys.save_data('frames.xyz')
        sys.profile_plot('statistics.pdf')

#r_correlation, v_correlation = Correlation(self.r_li, self.v_li)
#ax4.plot(np.linspace(0,t,len(r_correlation)), r_correlation, label='R') 
#ax4.set(title='Auto Correlation for displacement', xlabel='t', ylabel='<R(t)^2>')
#ax4.legend()
#ax5.plot(np.linspace(0,t,len(v_correlation)), v_correlation, label='C') 
#ax5.set(title='Auto Correlation for velocity', xlabel='t', ylabel='C(t)')
#ax5.legend()
