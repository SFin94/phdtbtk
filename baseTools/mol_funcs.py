"""Module containing function for processing and handling Molecule objecs."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xyz2mol import xyz2mol, read_xyz_file
from rdkit.Chem import AllChem as rdkit

import molLego as ml

# Global list of atoms - index matches ar
atom_id_index = ['h',  'he', 'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne', 'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar', 'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu']


def atom_id_to_index(atom_ids):
    """
    Convert list of atom ids to list of corresponding atom indexes - also implemented in molLego.

    Parameters
    ----------
    atom_ids: `list`
        Atom ids as str entry.

    Returns
    atom_indexes: `list`
        Atom indexes.

    """
    atom_indexes = [int(atom_id_index.index(i.lower()))+1 for i in atom_ids]

    return atom_indexes


def molecule_to_rdkit(molecule):
    """
    Convert a molLego Molecule to an rdkit Molecule using xyz2mol.
    
    Parameters
    ----------
    molecule: molLego `Molecule` object
        The Molecule to convert to rdKit
    
    Returns
    -------
    rdkit_mol: rdkit ``Molecule` object
        The rdKIT molecule.
    
    """
    # Set atom indexes if not already set 
    if hasattr(molecule, 'atom_indexes') == False:
        molecule.set_atom_indexes()

    # Use xyz2mol to create rdkit mol object
    rdkit_mol = xyz2mol(molecule.atom_indexes, molecule.geom)

    return rdkit_mol


def bonds_to_adjacency(bonds):
    """
    Construct an adjacency matrix from a tuple containing lists of connected bonds.
    
    Parameters
    ----------
    bonds: `tuple` of nested lists
        `list` of bond indexes for each atom in the molecule, order of lists matches the parent atoms index

    """
    # Initialise variables.
    mol_adjacency = np.zeros((len(bonds), len(bonds)))

    # Set bond entry in adjacency matrix for every bond in the connection list (non-pythonic index in list)
    for i, bond_i in enumerate(bonds):
        for j in bond_i:
            mol_adjacency[i, j-1] = 1

    return mol_adjacency


def recentre_dihedrals(dihedral_vals):
    """
    Recentre dihedral range: +/-180 to 0; 0 to +/-180.

    Parameters
    ----------
    dihedral_vals: :pandas:`Series`
        Values of dihedrals to be changed.
    
    Returns
    -------
    :pandas:`Series`
        Recentred dihedral values.

    """ 
    return dihedral_vals.apply(lambda x: x - (x/abs(x))*180)

