"""Module containing function for processing and handling Molecule objecs."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import networkx as nx

from xyz2mol import xyz2mol, read_xyz_file
import xyz2mol as xyz
from rdkit.Chem import AllChem as rdkit
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.rdmolfiles import MolToSmarts

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
    # Set atom indexes.
    molecule.set_atom_indexes()
    
    # Use xyz2mol to create rdkit mol object.
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
    mol_adjacency = GetAdjacencyMatrix(rdkit_mol)

    return mol_adjacency

# def test_molecule_indexes():

    # Find atom indexes (number and type)
    # Might need bond adjacency? - can use if molecule

    # Check is same or not

    # Check topology too
    
    # Do for two molecules or more? - in which case what format are the results?

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
    # i = 0
    # unique_atom = False
    # if reference_mol is None:
    #     reference_mol = molecules[0]
    # try:
    #     while unique_atom == False:
    #         start_atom = reference_mol.atom_ids[i]
    #         unique_atom = (reference_mol.atom_ids.count(start_atom) == 1)
    #         i += 1
    # except:
    #     print('No unique atom IDs to use for index start point.')
    #     raise
    
    # Set start node
    start_node = 0 
    
    for mol in molecules:
        # For now use adjacency here - as molecule is wrong then might not want to keep in future.
        adjacency = molecule_to_adjacency(mol)
        bonds = adjacency_to_bonds(adjacency)

        # Initialise graph.
        new = nx.Graph()
        new.add_edges_from(bonds)
        
        # Find all paths from start node by depth first search.
        path = [start_node]
        molecule_paths = []
        for edge in nx.dfs_edges(new, source=start_node):
            if edge[0] == start_node:
                molecule_paths.append(path)
                path = [edge[1]]
            else:
                path.append(edge[1])            
        molecule_paths.append(path)

        # Stack paths in order of length for new index list.
        new_index = molecule_paths.pop(0)
        while molecule_paths:
            path_lengths = [len(path) for path in molecule_paths]
            next_path = path_lengths.index(min(path_lengths))
            new_index.extend(molecule_paths.pop(next_path))
        
        # Reindex molecule.
        mol.reindex_molecule(new_index)

    return molecules


def adjacency_to_bonds(adjacency):
    """
    Compute bond list in molecule from ana adjacency matrix.

    Parameters
    ----------
    adjacency: :numpy:`array`
        Connectivty matrix.

    Returns
    -------
    bonds: :numpy:`array`
        List of bonds in terms of atom indexes.
    
    """
    bonds = np.nonzero(adjacency)
    
    return np.transpose(bonds)


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
        Recentered dihedral values.

    """ 
    return dihedral_vals.apply(lambda x: x - (x/abs(x))*180)


def calculate_dihedrals(molecules, dihedral_smarts):
    """
    Locate and calculate values for dihedrals in the molecules.

    Useful when the atom indexes aren't known and just the elements involves.
    Dihedral string needs to be just the four atom types.
        E.g. ['CNOP', 'CCOP']

    Parameters
    ----------
    molecules: `list` of :molLego:`Molecule`
        Molecules to calculate dihedrals for.
    dihedral_smarts: `dict`
        Key is dihedral string and value is dihedral SMARTS string.

    Returns
    -------
    molecules: `list` of :molLego:`Molecule`
        Molecules with dihedral values set.

    """
    # Calculate dihedral parameters for each molecule.
    for mol in molecules:

        # Find atom indexes for dihedrals.
        dihedral_indexes = {}
        for dihed, smarts in dihedral_smarts.items():

            # Convert molecule to rdkit molecule.
            rdkit_mol = molecule_to_rdkit(mol)

            # Find atom indexes for dihedral.
            query_mol = rdkit.MolFromSmarts(smarts)
            indexes = rdkit_mol.GetSubstructMatches(query=query_mol)
            
            # Save each set of indexes to dihedral dict.
            for j, ind in enumerate(indexes):
                dihedral_indexes[dihed + '_' + str(j)] = list(ind)

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
