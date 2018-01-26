import numpy as np

def write_head(fname, N, head):
    with open(fname, 'w') as f:
        f.write('{}\n'.format(N))
        f.write(head+'\n')

def write_atoms(fname, atom_coords, names=None):
    with open(fname, 'a') as f:
        for atom in atom_coords:
            if names==None:
                atom = ['C']+[str(i) for i in atom]
            line = ' '.join(atom)
            f.write('{}\n'.format(line))

def write_XYZ(fname, description, coordinates_arr):
    print('writing file {}'.format(fname))
    N = len(coordinates_arr)
    print('Particle number: {}\n description: {}'.format(N,description))
    write_head(fname, N, description)
    write_atoms(fname, coordinates_arr)
