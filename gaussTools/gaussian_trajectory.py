import sys
import numpy as np


def pull_atom_ids(input_file, atom_number):
    """
    Pull atom IDs from start of gaussian log file.
    
    Parameters
    ----------
    input_file: `str`
        Path/filename of input log file.
    atom_number: `int`
        Number of atoms in the molecule.

    Returns
    -------
    atom_ids: `list of str`
        Atom IDs of the molecule.

    """
    # Initialise variables.
    atom_ids = []

    # Set flags for searching for properties.
    atom_id_flag = 'Charge = '
    jump_line_flags = ['No Z-Matrix', 'Redundant internal coordinates']

    # Search for and parse atom number.
    with open(file_name, 'r') as input_file:
        for line in input_file:
            if atom_id_flag in line:

                # Get to start of input section and pull the atom ids from the list
                line = input_file.__next__()
                if any(flag in line for flag in jump_line_flags):
                    line = input_file.__next__()
                for atom in range(atom_number):
                    atom_ids.append(line.split()[0][0])
                    line = input_file.__next__()
                break
    return atom_ids

def pull_atom_number(input_file):
    """
    Pull the number of atoms from Gaussian Log file.

    Parameters
    ----------
    input_file: `str`
        Path/filename of input log file.

    Returns
    -------
    atom_number: `int`
        Number of atoms in the molecule.
    
    """
    atom_number = 0

    # Search for and parse atom number.
    with open(file_name, 'r') as input_file:
        for line in input_file:
            if 'natoms' in line.lower():
                atom_number = int(line.split()[1])
                break
    
    return atom_number

def pull_geometry(input_file, current_line, atom_number):
    """
    Pull cartesian coordinate geometry from standard orientation section in gaussian log file.

    Parameters
    ----------
    input_file: iterator
        Lines of file.
    current_line: `str`
        Current line of file.
    atom_number: `int`
        Number of atoms in the molecule.

    Returns
    -------
    atom_coords: :numpy:`array`
        xyz coordinates for each atom.
    
    """
    # Initalise variables.
    atom_coords = []

    # Skip the header section of the standard orientation block.
    [input_file.__next__() for x in range(0,4)]

    # Parse the atomic coordinates.
    for atom in range(atom_number):
        line = input_file.__next__()
        xyz = np.asarray([
            float(line.split()[i+3])
            for i in range(3)
        ])
        atom_coords.append(xyz)

    return np.asarray(atom_coords)

def push_geom_xyz(output_file, atom_number, atom_ids, trajectory):
    """
    Output molecule to an .xyz file.

    Parameters
    ----------
    output_file: `str`
        The name of the output xyz file.
    atom_number: `int`
        Number of atoms in the molecule.
    atom_ids: `list of str`
        Atom IDs of the molecule.
    trajectory: `list of :numpy:`array``
        xyz coordinates for each atom for each step in trajectory.
    
    """
    # Open output file, print header lines then atom indexes and cartesian coordinates to file
    with open(output_file + '.xyz', 'w+') as out_file:
        for geometry in trajectory:
            print(atom_number, file=out_file)
            print('Structure of {}'.format(output_file.split('.')[0]), file=out_file)
            for i, atom in enumerate(atom_ids):
                print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(atom, geometry[i]), file=out_file)
            print('',file=out_file)

if __name__ == "__main__":

    # Initialise variables.
    file_name = sys.argv[1]
    output_file = file_name.split('.')[0]
    geom_flag = 'Standard orientation'
    trajectory = []

    # Set atom number and atom IDs. 
    atom_number = pull_atom_number(file_name)
    atom_ids = pull_atom_ids(file_name, atom_number)
    
    # Pull each geometry from optimisation trajectory.
    with open(file_name, 'r') as input_file:
        for line in input_file:
            if geom_flag in line:
                trajectory.append(pull_geometry(input_file, line, atom_number))

    # Write optimisation trajectory to xyz file.
    push_geom_xyz(output_file, atom_number, atom_ids, trajectory)

