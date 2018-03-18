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

cut_off = lambda r: 1/2 - 1/2 * sin(pi/2 * (r-R)/D)
fR = lambda r: A * exp(-lam1 * r) 
fA = lambda r: B * exp(-lam2 * r) 
g_theta = lambda Ctheta: 1+ c**2 * ( 1/d**2 - 1/(d**2 + (h - Ctheta)**2) ) 
b_eta = lambda eta: (1 + (beta * eta)**n) ** (-1/(2*n))  


def g_func(rij, dij):
    dotijk = np.einsum('ijd,ikd->ijk', rij, rij) #j,k
    normijk = np.einsum('ij,ik->ijk', dij, dij)
    np.seterr(divide='ignore')
    Ctheta = dotijk / normijk
    return g_theta(Ctheta)

def b_func(rij, dij, cutij):
    etaijk = np.einsum('ij, ijk->ijk', cutij, g_func(rij, dij) ) 
    s0,s1,s2 = etaijk.shape
    etaij = np.sum(etaijk, axis=1) - etaijk.reshape(s0,-1)[:,::s2+1] #diagonal
    return b_eta(etaij)

def group_by_neighbors(cut_groups):
    length_d = {}
    for neighbors in cut_groups:
        N = neighbors.size 
        if N not in length_d:
            length_d[N] = neighbors[None, :]
        else:
            length_d[N] = np.vstack((length_d[N], neighbors))
    return length_d

def neighborV_gen(rjN, djN, cut_groups):
    cutj =  np.where( djN>R-D, cut_off(djN), 1)

    yield np.sum( cutj * fR(djN) )

    fAj = fA(djN)
    length_d = group_by_neighbors(cut_groups)
    for N,inds in length_d.items(): 
        if N == 1:    
            ind1 = inds[:, 0]
            yield np.sum( -cutj[ind1] * fAj[ind1] ) 
        else:
            bij = b_func(rjN[inds], djN[inds], cutj[inds])  
            yield np.sum( cutj[inds] * bij * -fAj[inds])
        
    #for js in cut_groups:
    #    if len(js) == 1:
    #        bj = 1
    #    else:
    #        bj = b_func(rjN[js], djN[js], cutj[js])  
    #    yield np.sum( -cutj[js] *  bj * fAj[js]) 
        
def tersoff_V(pos):
    natoms = len(pos) 

    rij = pos[None, :] - pos[:, None] + np.identity(natoms)[:,:,None]
    dij =  np.linalg.norm( rij, axis=2 )
    dij+=  np.identity(natoms) * (2 * R)

    cut_mask = dij < R+D
    if not cut_mask.any(): return 0.
    cut_ind = np.where(cut_mask)
    cut_groups = np.split(np.arange(len(cut_ind[0])),
        np.where( np.diff(cut_ind[0]) )[0]+1)

    return sum( neighborV_gen(rij[cut_ind], dij[cut_ind], cut_groups) ) / 2 


def tersoff_F(pos):
    dEdR = elementwise_grad(tersoff_V)
    return -dEdR(pos)

