'''This script no longer applies'''
import numpy as np

def create_atoms(names, N, scale):
    atom_coords = []
    for i in range(N):
        atom_name = [np.random.choice(names)]
        coords = [str(i) for i in scale * np.random.random(3)]
        atom = atom_name + coords
        atom_coords.append(atom)
    return atom_coords

def write_atoms(f, atom_coords):
    for atom in atom_coords:
        line = ' '.join(atom)
        f.write('{}\n'.format(line))

def write_frames(fname):
    names = ['C','H','N']
    N = 8
    scale = 2
    for i in range(3):
        with open(fname, 'a') as f:
            f.write( f'{N}\n' )
            f.write( f'frame {i}\n')
            atom_coords = create_atoms(names, N, scale) 
            write_atoms(f, atom_coords)
    
write_frames('frames.xyz') 
