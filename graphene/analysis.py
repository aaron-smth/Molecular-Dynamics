import numpy as np
from numpy import pi
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks_cwt 
from scipy import signal
import matplotlib.pyplot as plt 
import pickle
from MD import MD_sys
from tersoff import tersoff_V
from itertools import groupby
import pandas as pd

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

def plot_peaks(xs, ys, ax, width=15):
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

def tau_pot(rijd):
    from tersoff import tersoff_pairwise_F
    Fijd = tersoff_pairwise_F(rijd) 
    return np.einsum('ija, ijb -> ab', rijd, Fijd) / 2

def tau_kin(vid, m):
    v_mean = np.mean(vid, axis=0)
    centered_v = vid - v_mean
    return np.einsum('ia, ib -> ab', centered_v, centered_v) * (- m)
    
def Virial_Stress(pos, vel, m, L):
    natoms = len(pos) 
    rijd = pos[np.newaxis, :, :] - pos[:, np.newaxis, :] + np.eye(natoms)[:,:,None]
    Volume = L**2 * 3.45 # the thickness of a graphene sheet is 3.45 A
    return (tau_pot(rijd) + tau_kin(vel, m))/ Volume

def particle_track(n_probes, r_li, fixed_ind):
    free_inds = np.delete(np.arange(len(r_li[0])), fixed_ind)
    inds = np.random.choice(free_inds, size=n_probes)
    chosen_pos = np.array([r[inds, 2] for r in r_li])
    probe_arr = np.einsum('tn->nt', chosen_pos) 
    # (time, nth particle, direction) -> (nth particle, direction, time)
    return probe_arr, inds


