import autograd.numpy as np
from autograd import elementwise_grad
from autograd.numpy import sqrt, log, sin, cos, pi, exp

# parameters
A = 1393.6 #ev
B = 430. #old:346.74 
lam1 = 3.4879 #A^-1
lam2 = 2.2119
lam3 = 0 
n = 0.72751 #1
c = 38049.0
d = 4.3484
h = -0.930 #old:-0.57058
beta= 1.5724e-7
R = 1.95 #A
D = 0.15

def b_func(rj, dj, cutj):
    etajk = np.einsum('k, jk->jk', cutj, g_func(rj, dj) ) 
    etaj = np.sum(etajk, axis=1) - np.diag(etajk)
    return b_eta(etaj)

cut_off = lambda r: 1/2 - 1/2 * sin(pi/2 * (r-R)/D)
f_R = lambda r: A * exp(-lam1 * r) 
f_A = lambda r: B * exp(-lam2 * r) 
g_theta = lambda Ctheta: 1+ c**2 * ( 1/d**2 - 1/(d**2 + (h - Ctheta)**2) ) 
b_eta = lambda eta: (1 + (beta * eta)**n) ** (-1/(2*n))  

def g_func(rj, dj):
    natoms = len(rj)
    dotjk = np.einsum('jd,kd->jk', rj, rj) #j,k
    norm = np.einsum('j,k->jk', dj, dj)
    norm += np.identity(natoms)
    Ctheta = dotjk / norm  

    return g_theta(Ctheta)

def neighborV_gen(rijd, dij, cut_ind):
    natoms = len(dij)
    for i in range(natoms):
        js = cut_ind[1][ cut_ind[0] == i ]
        if len(js) == 0:
            yield 0

        rj  = rijd[i, js]
        dj = dij[i, js]
        cutj = np.where( dj>R-D, cut_off(dj), 1)
        
        if len(js) == 1:
            bj = 1
        else:
            bj = b_func(rj, dj, cutj) 
        yield np.sum( cutj * (f_R(dj) - bj * f_A(dj)) )
        

def tersoff_V(pos):
    natoms = len(pos) 

    rijd = pos[None, :] - pos[:, None] + np.identity(natoms)[:,:,None]
    dij =  np.linalg.norm( rijd, axis=2 )
    dij += np.identity(natoms)*2*R

    cut_ind = np.where(dij < R+D)
 
    return sum( neighborV_gen(rijd, dij, cut_ind) ) / 2 

def tersoff_F(pos):
    dEdR = elementwise_grad(tersoff_V)
    return -dEdR(pos)
