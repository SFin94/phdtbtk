"""Module containing general functions for phd processing and tools."""

import sys
import numpy as np
import pandas as pd

import molLego as ml


def process_input_file(input_file):
    """
    Process input file creating DataFrame of molecule information and/or Molecule/MoleculeThermo objects.
    
    Parameters
    ----------
    input_file: `str`
        The input file name.
        Should be one of .log, .xyz, .conf or .csv.

    Returns
    -------
    mol_df: `pd DataFrame`
        A dataframe of molecule information.
    molecules: `list of Molecule/MoleculeThermo objects`
        The Molecule objects for each input molecule if .log, .xyz or .conf.
        Otherwise None type if .csv.

    """
    # Retrieve file type for input file
    file_type = str(input_file.split('.')[-1])

    # Process Molecule objects and DataFrame if direct input source
    if file_type == 'conf':
        mol_names, mols = ml.construct_mols(input_file)
        mol_df = ml.mols_to_dataframe(mols, mol_names=mol_names)

    elif file_type == 'log':
        mols = ml.init_mol_from_log(input_file)
        mol_df = ml.mols_to_dataframe(mols)

    elif file_type == 'xyz':
        mols = ml.init_mol_from_xyz(input_file)
        mol_df = ml.mols_to_dataframe(mols)

    # Parse existing dataframe and set first column (mol_names) as index
    elif file_type == 'csv':
        mol_df = pd.read_csv(input_file, index_col=0) 
        mols = None
        
    else:
        raise Exception('Input file is not a compatiable file type (.log, .xyz, .conf or .csv)')

    return mol_df, mols


def readlines_reverse(input_file):
    """
    Read lines of a file in reverse order [edited from Filip Szczypiński].

    Parameters
    ----------
    input_file: `str`
        The file to be read

    """ 
    with open(input_file) as in_file:
        
        # Move to end of file
        in_file.seek(0, 2)
        position = in_file.tell()
        line = ''

        # Iterate over lines until top is reached
        while position >= 0:
            in_file.seek(position)
            next_char = in_file.read(1)
            if next_char == "\n":
                yield line[::-1]
                line = ''
            else:
                line += next_char
            position -= 1
        yield line[::-1]

def clean_string(dirty_string):
    """
    Remove symbols from string.

    Parameters
    ----------
    dirty_string: `str`
        String to have symbols removed from.

    Returns
    -------
    clean_string: `str`
        String with only alpha/numeric characters.

    """
    clean_string = ''
    for char in dirty_string:
        if char.isalnum():
            clean_string += char
    
    return clean_string

def push_geom_xyz(output_file, molecule):
    """
    Output molecule to an .xyz file.

    Parameters
    ----------
    output_file: `str`
        The name of the output xyz file.
    molecule: `Molecule object`
        The molecule to be output.
    
    """
    # Open output file, print header lines then atom indexes and cartesian coordinates to file
    with open(output_file + '.xyz', 'w+') as out_file:
        print(molecule.atom_number, file=out_file)
        print('Structure of {}'.format(output_file.split('.')[0]), file=out_file)
        for atom_ind, atom in enumerate(molecule.atom_ids):
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(atom, molecule.geom[atom_ind]), file=out_file)

def to_smarts(parameter):
    """
    Convert an atom id based parameter string to smarts code.

    Parameters
    ----------
    parameter: `str`
        The atom ID string for the parameter.
        E.g. POCC for the P-O-C-C dihedral

    Returns
    -------
    smarts: `str`
        The Smarts string for the parameter.

    """
    # Intialise smarts string.
    atom_id = parameter[0]
    smarts = '['
    
    # Insert [] characters around every atom ID.
    for p in parameter[1:]:
        if p.isupper():
            if atom_id[-1] != '=':
                smarts += atom_id+']'
            else:
                smarts += atom_id
            atom_id = '[' + p
        elif p == '=':
            atom_id += ']='
        else:
            atom_id += p
    # Append final entry.
    smarts += (atom_id+']')

    return smarts


def from_smarts(smarts):
    """
    Convert an atom id based parameter string to smarts code.

    Parameters
    ----------
    smarts: `str`
        The Smarts string for the parameter.

    Returns
    -------
    parameter: `str`
        The atom ID string for the parameter.
        E.g. POCC for the P-O-C-C dihedral

    """
    parameter = smarts.replace('[', "")
    parameter = parameter.replace(']', "")

    return parameter
