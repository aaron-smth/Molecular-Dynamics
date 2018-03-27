import numpy as np
from scipy import signal
import pylab as plt
from itertools import chain

def L_map(r, x):
    return np.longdouble(4) * r * x * (1 - x)

x0 = np.longdouble(0.6)
#rs = np.linspace(0.7,1,2000, dtype=np.longfloat)
rs  =  np.linspace(0.872 , 0.877 , 10000, dtype=np.longdouble)

def x_gen(rs, x0, steps):
    cut = 1000
    x = np.ones(rs.size) * x0
    for i in range(steps):
        x = L_map(rs, x)
        if i > cut:
            yield x

def Bifurcate_plot():
    fig, ax = plt.subplots(figsize=(30,16)) 
    xs = np.vstack( list(x_gen(rs, x0,2000)) )
    ax.scatter(np.repeat(rs, xs.shape[0], axis=0) ,xs.T, s=1, c='black', alpha=0.5)
    ax.set(xlabel='r', ylabel='x')
    fig.savefig('bif.png') 

#np.seterr(divide='raise')
def Tdivide(a,b):
    c = a/b
    abnormal = np.isnan(c) | (c==0)
    c[abnormal] = 1.
    return c, abnormal

def Ly_exponent(x0, steps):
    x = np.ones(rs.size) * x0
    y = x + np.longdouble(1.e-8)

    diff_memory = np.abs(y-x)
    lams = 0
    abnormals = np.zeros(rs.size)
    for i in range(steps):
        x = L_map(rs, x)
        y = L_map(rs, y)
        diff = np.abs(y-x)
        divided, abnormal = Tdivide(diff, diff_memory)
        abnormals += abnormal
        lams += np.log( divided )
        diff_memory = diff
    lams /= (steps - abnormals)
    print('loss:', (abnormals / steps ).mean() )
    #print('abnormal rs, lams:',rs[lams>-0.01], lams[lams>-0.01]) 
    return lams, abnormals

def Ly_plot(rs, lams, ax1, ax2):
    ax1.plot(rs, lams, label='raw')
    ax1.set(xlabel='r', ylabel='$\lambda$')

    sp = np.fft.fft(lams)
    freq = np.fft.fftfreq(lams.shape[-1])
    ax2.plot(freq, np.abs(sp)**2)
    ax2.set(xlabel='frequency', ylabel='Signal Power')


def Ly_mask_compare(lams):
    fig, axes = plt.subplots(2, figsize=(30,16))
    ax1, ax2 = axes 
    ax2.set(xscale='log', yscale='log')

    mask = lams < -0.02

    Ly_plot(rs, lams, ax1, ax2)
    #Ly_plot(rs[mask], lams[mask], ax1, ax2) 

    fname = f'ly_{(round(rs[0],3), round(rs[-1], 3))}.png'
    fig.savefig(fname)

def traj_gen(rs, x0, steps):
    x = np.ones(rs.size, dtype=np.longdouble) * x0
    for i in range(steps):
        x = L_map(rs, x)
        yield x

def trajectory(rs):  
    fig, ax = plt.subplots(figsize=(30,16))
    xs = np.vstack(list(traj_gen(rs, x0, 200)))
    ys = np.vstack(list(traj_gen(rs, x0+1.e-10, 200))) 
    ax.set(yscale='log', xlabel='steps', ylabel='$|\Delta x|$')
    for i,r in enumerate(rs):
        ax.plot(np.abs(xs[:, i]-ys[:, i]), label=f'r:{r}')
    fig.savefig('trajectory.png')

def abnormal_plot(lams, abnormals):
    fig, axes = plt.subplots(2, figsize=(30,16))    
    ax1, ax2 = axes
    mask1 = abnormals > 0 
    ax1.scatter(lams[mask1], abnormals[mask1], s=10)
    ax1.set(xlabel='$\lambda$', ylabel='skipped points')
    mask2 = lams > -0.1
    ax2.scatter(lams[mask1 & mask2], abnormals[mask1 & mask2], 
            label='points with big $lambda$')
    ax2.set(xlabel='$\lambda$', ylabel='skipped points')
    fig.savefig('abnormal.png')

plt.rcParams.update({'font.size': 22})
#Bifurcate_plot()
lams, abnormals = Ly_exponent(x0, 3000)
#Ly_mask_compare(lams)
traj_rs = rs[lams>-0.02][:10]
trajectory(traj_rs)
np.savetxt('traj_rs.out', traj_rs, delimiter=',')
#abnormal_plot(lams, abnormals)



