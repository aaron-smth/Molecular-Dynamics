import numpy as np
import os

def to_xyz(f, description, arr, name='Ar', dfmt='%.5f'): 

    N, dim = arr.shape 

    '''writing head'''
    head = np.array([[str(N)],[description]])

    '''remember to use write binary mode '''
    np.savetxt(f, head, fmt='%s')
    np.savetxt(f, arr,  fmt=f'{name}' + f' {dfmt}'*dim)

def from_xyz(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    N = int(lines[0])
    pos = lines[-N:]
    pos = [line[:-1].split(' ') for line in pos] 
    arr = np.array( pos )
    arr = arr[:,1:].astype(float)
    return arr
