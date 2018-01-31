'''This script no longer applies'''
import numpy as np

def write_head(fname, N, head):
    print('Writting to file {}'.format(fname))
    with open(fname, 'w') as f:
        f.write('{}\n'.format(N))
        f.write(head+'\n')

def create_atoms(names, N, scale):
    atom_coords = []
    for i in range(N):
        atom_name = [np.random.choice(names)]
        coords = [str(i) for i in scale * np.random.random(3)]
        atom = atom_name + coords
        atom_coords.append(atom)
    return atom_coords

def write_atoms(fname, atom_coords):
    with open(fname, 'a') as f:
        for atom in atom_coords:
            line = ' '.join(atom)
            f.write('{}\n'.format(line))

def write_frames(frame):
    fname = 'lab0_frame{}.xyz'.format(frame)
    names = ['C','H','N']
    N = 8
    scale = 2
    write_head(fname, N, 'lab0_2 example frame {}'.format(frame))
    atom_coords = create_atoms(names, N, scale)
    write_atoms(fname, atom_coords)
    
for i in range(3):
    write_frames(i)
    
