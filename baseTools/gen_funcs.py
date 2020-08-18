import sys
import numpy as np
import pandas as pd

import molLego as ml


def process_input_file(input_file):

    '''Processes the input file, if a conf file is given then the molecules are processed, creating Molecule/MoleculeThermo objects for each enetry in the conf file and converting to a DataFrame. If a csv file is given then the molecule DataFrame is parsed directly from the csv file
    
    Parameters:
     input_file: str - file name which should have either a .conf or .csv extension

    Returns:
     mol_df: pd DataFrame - dataframe of molecule information
     [optional returns if .conf file is the input file type]
     molecules: list of Molecule/MoleculeThermo objects - created Molecule objects for each entry line in the conf file

    '''
    
    # Retrieve file type for input file
    file_type = str(input_file.split('.')[-1])

    # Process conf file, creating Molecule objects and a DataFrame
    if file_type == 'conf':

        mol_names, mols = ml.construct_mols(input_file)
        mol_df = ml.mols_to_dataframe(mols, mol_names=mol_names)
        return mol_df, mols

    elif file_type == 'xyz':
        mols = ml.init_mol_from_xyz(input_file)
        mol_df = ml.mols_to_dataframe(mols)
        return mol_df, mols

    # Parse in existing dataframe and set first column (mol_names) as index
    elif file_type == 'csv':
        mol_df = pd.read_csv(input_file, index_col=0) 
        return mol_df, None
    
    else:
        return


def parse_xyz(xyz_file):

    '''
    Parses geometry and atom ids from an xyz file
    
    Parameters:
     xyz_file: str - name/path of input xyz file

    Returns:
     geometry: np array - x, y, z coordinates for each atom
     atom_ids: list of str - atom id of each atom
    
    '''
    # Open xyz file
    with open(xyz_file, 'r') as input:
        atom_number = int(input.readline())
        for line in input:

            # If multiple molecules then process next one with new atom number
            if len(molecule) >= 1:
                atom_number = int(line.split()[0])
                line = input.__next__()

            # Intialise variables
            atom_ids = []
            atom_coords = []
            
            # Pull geometry and atom ids
            for i in range(atom_number):
                line = input.__next__()
                atom_ids.append(line.split()[0])
                xyz = np.asarray([float(i) for i in line.split()[1:]])
                atom_coords.append(xyz)
            geometry = np.asarray(atom_coords)

    return geometry, atom_ids