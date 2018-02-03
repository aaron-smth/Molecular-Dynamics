import numpy as np
import os

def to_xyz(f, description, arr, name='Ar', dfmt='%.5f'): 
    if len(arr.shape) == 1:
        arr = np.array([arr])

    N, dim = arr.shape 

    '''writing head'''
    head = np.array([[str(N)],[description]])

    '''remember to use write binary mode '''
    np.savetxt(f, head, fmt='%s')
    np.savetxt(f, arr,  fmt=f'{name}' + f' {dfmt}'*dim)


