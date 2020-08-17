import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xyz2mol import xyz2mol, read_xyz_file
from rdkit.Chem import AllChem as rdkit

import molLego as ml

'''Script containing functions for processing Molecule objecs and repeated analysis/processing tasks'''

# Global list of atoms - index matches ar
atom_id_index = ['h',  'he', 'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne', 'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar', 'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu']


def atom_id_to_index(atom_ids):

    '''Function to convert list of atom ids to list of corresponding atom indexes - also implemented in molLego

    Parameters:
     atom_ids: list - atom ids as str entry

    Returns:
     atom_indexes: list - atom indexes
    '''

    atom_indexes = [int(atom_id_index.index(i.lower()))+1 for i in atom_ids]

    return atom_indexes


def molecule_to_rdkit(molecule):

    '''Function that converts a Molecule object to an rdkit molecule using xyz2mol
    
    Parameters:
     molecule: Molecule object
    
    Returns:
     rdkit_mol: rdkit Molecule object
    '''
    
    # Set atom indexes if not already set 
    if hasattr(molecule, 'atom_indexes') == False:
        molecule.set_atom_indexes()

    # Use xyz2mol to create rdkit mol object
    rdkit_mol = xyz2mol(molecule.atom_indexes, molecule.geom)

    return rdkit_mol

