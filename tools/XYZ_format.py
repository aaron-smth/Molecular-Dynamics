import numpy as np
import os

def write_head(fname, N, head):
    with open(fname, 'w') as f:
        f.write(f'{N}\n')
        f.write(head+'\n')

def write_atoms(fname, atom_coords, names=None):
    with open(fname, 'a') as f:
        for atom in atom_coords:
            if names==None:
                atom = ['C']+[str(i) for i in atom]
            line = ' '.join(atom)
            f.write(f'{line}\n'

def write_XYZ(fname, description, coordinates_arr):
    head = './XYZ/'
    os.makedirs(head, exist_ok=True)

    print('writing file {fname}')

    N = len(coordinates_arr)
    print('Particle number: {N}\n description: {description}')

    write_head(head + fname, N, description)
    write_atoms(head + fname, coordinates_arr)
