import numpy as np
import os

def to_xyz(f, description, arr, atom_name='Ar'): 

    N, dim = arr.shape 

    '''writing head'''
    head = np.array([[str(N)],[description]])

    '''remember to use write binary mode '''
    np.savetxt(f, head, fmt='%s')
    np.savetxt(f, arr, fmt= atom_name+' %.6f'*dim)

