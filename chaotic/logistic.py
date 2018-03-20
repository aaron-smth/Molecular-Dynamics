import numpy as np
import pylab as plt
from itertools import chain

def L_map(r, x):
    return 4 * r * x * (1 - x)

x0 = 0.6
rs = np.linspace(0.7, 1, 1500)

def x_gen(x0, steps):
    rs = np.linspace(0.7, 1, 1500)
    cut = 1000
    x = np.ones(rs.size) * x0
    for i in range(steps):
        x = L_map(rs, x)
        if i > cut:
            yield x

def Bifurcate_plot():
    fig, ax = plt.subplots(figsize=(30,16)) 
    xs = np.vstack( list(x_gen(x0,2000)) )
    ax.scatter(np.repeat(rs, xs.shape[0], axis=0) ,xs.T, s=1, c='b')
    fig.savefig('bif.png') 

#np.seterr(divide='raise')
def Tdivide(a,b):
    c = a/b
    abnormal = np.isnan(c) | (c==0)
    c[abnormal] = 1.
    return c, abnormal

def Ly_exponent(x0, steps):
    x = np.ones(rs.size) * x0
    y = x + 1e-10

    diff_memory = np.abs(y-x)
    lam = 0
    abnormals = np.zeros(rs.size)
    for i in range(steps):
        x = L_map(rs, x)
        y = L_map(rs, y)
        diff = np.abs(y-x)
        divided, abnormal = Tdivide(diff, diff_memory)
        abnormals += abnormal
        lam += np.log( divided )
        diff_memory = diff
    lam /= (steps - abnormals)
    return lam

def Ly_plot():
    lams = Ly_exponent(x0, 2000) 

    fig, ax = plt.subplots(figsize=(30,16))
    ax.plot(rs, lams)
    fig.savefig('ly_exp.png')


Bifurcate_plot()
#Ly_plot()
