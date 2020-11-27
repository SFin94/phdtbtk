"""Module containing function for processing and handling Molecule objecs."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from xyz2mol import xyz2mol, read_xyz_file
from rdkit.Chem import AllChem as rdkit
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolops import RDKFingerprint

import molLego as ml
import phdtbtk.baseTools.gen_funcs as gen_funcs

# Global list of atoms - index matches ar
__ATOM_TYPES__ = ['h',  'he', 'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne', 'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar', 'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu']


def atom_type_to_number(atom_types):
    """
    Convert atom types to corresponding atom number (atomic number).

    Also implemented in molLego.

    Parameters
    ----------
    atom_types: `list of str`
        Atom type for each atom.

    Returns
    -------
    atom_numbers: `list of int`
        Atomic number for each atom.

    """
    atom_numbers = [int(__ATOM_TYPES__.index(i.lower()))+1 for i in atom_types]

    return atom_numbers


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


def molecule_to_adjacency(molecule):
    """
    Construct an adjacency matrix from a tuple containing lists of connected bonds.
    
    Parameters
    ----------
    molecule: :molLego:`Molecule`
        Molecule to calculate adjacency matrix for.

    """
    # Convert molecule to rdkit mol.
    rdkit_mol = molecule_to_rdkit(molecule)

    # Calculate adjacency matrix.
    mol_adjacency = GetAdjacencyMatrix(rdkit_mol, useBO=True)

    return mol_adjacency

# def test_molecule_indexes():

    # Find atom indexes (number and type)
    # Might need bond adjacency? - can use if molecule

    # Check is same or not

    # Check topology too
    
    # Do for two molecules or more? - in which case what format are the results?


def find_paths(current_step, adjacency, prev_step, reindex_map):
    """
    Construct a branch of a reaction path.

    Parameters
    ----------
    current_step: `int`
        Current atom index of path.
    adjacency: :numpy:`array`
        Connecitivty matrix.
        Entries are 1 for connected points or 0 for unconnected points.
    prev_step: `int`
        Previous atom index in path.
    reindex_map: `list`
        Mapping of old index to new position.

    Returns
    -------
    reindex_map: `list`
        Mapping of old index to new position.
    
    """
    # Locate connecting reaction steps for current paths.
    next_steps = np.nonzero(adjacency[current_step,:])[0]
    for step in next_steps:
        if step != prev_step and step not in reindex_map:
            reindex_map.append(step)
            reindex_map = find_paths(step, adjacency, current_step, reindex_map)
    return reindex_map


def match_indexes(molecules, reference_mol=None):
    """
    Match indexes of one molecule to another.

    Parameters
    ----------
    molecule1: :molLego:`Molecule`
        Molecule with reference atom indexes.
    molecule2: :molLego:`Molecule`
        Molecule to reindex.
    
    Returns
    -------
    molecule2: :molLego:`Molecule`
        Reindexed Molecule.

    """
    # Find unique atom type to start.
    i = 0
    unique_atom = False
    if reference_mol is None:
        reference_mol = molecules[0]
    try:
        while unique_atom == False:
            start_atom = reference_mol.atom_ids[i]
            unique_atom = (reference_mol.atom_ids.count(start_atom) == 1)
            i += 1
    except:
        print('No unique atom IDs to use for index start point.')
        raise
    
    for mol in molecules:
        adjacency = molecule_to_adjacency(mol)
        start_index = mol.atom_ids.index(start_atom)
        reindex_map = [start_index]
        start_nodes = np.nonzero(adjacency[start_index,:])[0]
        for node in start_nodes:
            reindex_map.append(node)
            reindex_map = find_paths(node, adjacency, start_index, reindex_map)
        print(reindex_map)
        mol.reindex_molecule(reindex_map)
    
    return molecules


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


def calculate_dihedrals(molecules, dihedrals):
    """
    Locate and calculate values for dihedrals in the molecules.

    Useful when the atom indexes aren't known and just the elements involves.
    Dihedral string needs to be just the four atom types.
        E.g. ['CNOP', 'CCOP']

    Parameters
    ----------
    molecules: `list` of :molLego:`Molecule`
        Molecules to calculate dihedrals for.
    dihedrals: `list`
        Dihedral strings of atom types.

    Returns
    -------
    molecules: `list` of :molLego:`Molecule`
        Molecules with dihedral values set.

    """
    # Calculate dihedral parameters for each molecule.
    for mol in molecules:

        # Convert molecule to rdkit mol.
        rdkit_mol = molecule_to_rdkit(mol)

        # Find atom indexes for dihedrals.
        dihedral_indexes = {}
        for i, dihed in enumerate(dihedrals):

            # Convert dihedral to smarts and find indexes in molecule.
            dihedral_smarts = gen_funcs.to_smarts(dihed)
            query_mol = rdkit.MolFromSmarts(dihedral_smarts)
            indexes = rdkit_mol.GetSubstructMatches(query=query_mol)
            
            # Save each set of indexes to dihedral dict.
            dihed_key = gen_funcs.clean_string(dihed)
            for j, ind in enumerate(indexes):
                dihedral_indexes[dihed_key + '_' + str(j)] = list(ind)

        # Calculate dihedral values.
        mol.set_parameters(dihedral_indexes)

    return molecules

def calculate_conformer_RMSD(molecules):
    """
    Calculate RMSD with lowest energy molecule.

    Parameters
    ----------
    molecules: `list` of :molLego:`Molecule`
        Molecules to calculate RMSD for.
    
    Returns
    -------
    rmsd: `list`
        RMSD values for each molecule with the lowest energy molecule.
        
    """
    # Initalise reference mol and energy from first molecule.
    lowestE = molecules[0].escf
    reference_mol = molecule_to_rdkit(molecules[0])

    # Search for lowest energy molecule.
    for mol in molecules:
        if mol.escf < lowestE:
            reference_mol = molecule_to_rdkit(mol) 

    # Calculate RMSD with lowest E conformer for each conformer.
    rmsd = []
    for mol in molecules:
        # Convert molecule to rdkit mol.
        rdkit_mol = molecule_to_rdkit(mol)
        try:
            rmsd.append(AlignMol(rdkit_mol, reference_mol))
        except:
            rmsd.append(np.nan)
    
    return rmsd
