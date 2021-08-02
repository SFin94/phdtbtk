"""Module containing function for processing and handling Molecule objecs."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import networkx as nx

from xyz2mol import xyz2mol, read_xyz_file, get_AC
import xyz2mol as xyz
from rdkit.Chem import AllChem as rdkit
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.rdmolfiles import MolToSmarts

import molLego as ml
import phdtbtk.base_tools.gen_funcs as gen_funcs

# Global list of atoms - index matches ar
__ATOM_TYPES__ = ['h',  'he', 'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne', 'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar', 'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu']

def process_mol_input(file_name, parser=ml.GaussianLog, 
                      mol_type=ml.GaussianThermoMolecule,
                      as_dataframe=False):
    """
    Create Molecules or molecule DataFrame from input file.

    Input file can be *.csv containing previously processed molecule
    data. Or *.conf containing list of molecules to be processed. For
    a *.conf lists of molecule names and Molecule objects will be 
    returned unless as_dataframe is set to ``True``.

    Parameters
    ----------
    file_name : :class: `str`
        Name/path of molecule input file to be processed.

    parser : :OutputParser:
        Parser class to use for calculation output.

    molecule_type : :Molecule:
        Molecule class to use for calculation output.
    
    as_dataframe : :class:`bool`
        [Default:``False``] If True returns data frame.
        If False returns molecules and molecule names.

    Returns
    -------
    molecules : :class:`list` of :Molecule:
        Molecule objects for each file in system conf file.

    """
    if file_name.split('.')[-1] == 'csv':
        return pd.read_csv(file_name, index_col=0)

    elif file_name.split('.')[-1] == 'conf':
        mol_names, mols = ml.construct_mols(file_name, parser=parser, molecule_type=mol_type)
        if as_dataframe == False:
            return mol_names, mols
        else:
            return ml.mols_to_dataframe(mols, mol_names=mol_names)


def get_neighbour_list(file_name):
    """
    Get neighbour list only from .conf file of reaction system.

    Example formatting for a reaction: A + B --> C --> D + E
        reactants A_output[.ext],B_output[.ext] int
        int C_output[.ext] products
        products D_output[.ext],E_output[.ext] 
        
    Where [.ext] must be compatiable with the parser specified.
    Lines can be commented out with leading '#'.

    Parameters
    ----------
    file_name : :class: `str`
        Name/path of molecule input file to be processed.

    Returns
    -------
    neighbour_indexes : :class:`list` of `int`
        Neighbour list of reaction steps.

    """
    # Initialise variables
    mol_names = []
    molecules = []
    step_neighbours = []

    # Process files and names in system conf file.
    with open(system_file, 'r') as infile:
        for system_line in infile:
            if system_line[0] != '#':

                # Set reaction step names and molecules.
                raw_in = system_line.split()
                mol_names.append(raw_in[0])
                
                # Set neighbour list.
                if len(raw_in) > 2:
                    step_neighbours.append(raw_in[2].split(','))
                else:
                    step_neighbours.append([])

    # Convert step neighbours to reaction step indexes.
    neighbour_indexes = []
    for step in step_neighbours:
        step_indexes = []
        for i in step:
            step_indexes.append(mol_names.index(i))
        neighbour_indexes.append(step_indexes)

    # Initialise reaction from system.
    return neighbour_indexes

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
    molecule_atomic = list(molecule.get_atomic_numbers())

    # Use xyz2mol to create rdkit mol object.
    rdkit_mol = xyz2mol(molecule_atomic, molecule.geometry, charge=molecule.charge)
    
    return rdkit_mol

def molecule_to_adjacency(molecule):
    """
    Construct an adjacency matrix using RDKit.
    
    Parameters
    ----------
    molecule: :molLego:`Molecule`
        Molecule to calculate adjacency matrix for.

    """
    # Convert molecule to rdkit mol.
    rdkit_mol = molecule_to_rdkit(molecule)

    # Calculate adjacency matrix.
    return GetAdjacencyMatrix(rdkit_mol)
    # return get_AC(rdkit_mol, covalent_factor=1.35)

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
    # Find all non zero entries and return as tuples.
    return np.transpose(np.nonzero(adjacency))


def find_paths(start_node, molecule, mol_graph):
    """Find paths in graph from start node by depth first search."""
    path = [start_node]
    molecule_paths = []
    for edge in nx.dfs_edges(mol_graph, source=start_node):
        if edge[0] == start_node:
            molecule_paths.append(path)
            path = [edge[1]]
        else:
            path.append(edge[1])
    molecule_paths.append(path)
    return molecule_paths

def stack_paths(molecule_paths):
    """Set index list of molecule paths from longest to shortest."""
    new_index = []
    # Stack paths in order of length for new index list.
    while molecule_paths:
        path_lengths = [len(path) for path in molecule_paths]
        next_path = path_lengths.index(min(path_lengths))
        new_index.extend(molecule_paths.pop(next_path))
    return new_index

def index_by_paths(molecules, reference_mol=None, start_node=0):
    """
    Reindex a molecule using order of paths from a starting node.

    Parameters
    ----------
    molecules: `list of :molLego:`Molecule``
        List of Molecules to be reindexed.
    reference_mol: :molLego:`Molecule`
        Moelcule to match indexes to. [Default: None type]
        If `None` then defaults to first molecule in molecules list.
    start_node: `int`
        The index (0 index) of the starting atom to use in the 
        reference molecule. 
        Must be a unique atom type or the same in all molecules.
    
    Returns
    -------
    molecules: `list of :molLego:`Molecule``
        List of reindexed Molecules.

    """
    # Set reference molecule as first molecule in list if not set.
    if reference_mol is None:
        reference_mol = molecules[0]

    # Find unique atom type for start node if not set.
    if start_node is not None:
        i = 0
        unique_atom = False
        try:
            while unique_atom == False:
                start_atom = reference_mol.atom_ids[i]
                unique_atom = (reference_mol.atom_ids.count(start_atom) == 1)
                i += 1
        except:
            print('No unique atom IDs to use for index start point.')
            raise
    else:
        start_atom = reference_mol.atom_ids[start_node]
        try: 
            reference_mol.atom_ids.count(start_atom) == 1
        except:
            print('Warning, starting atom is not a unique type. \
            Assuming same index in all molecules')

    # Reindex each molecule from starting node.
    for mol in molecules[:2]:
        print('start ids:')
        print(mol.atom_ids)
        # Calculate adjacency matrix and bond list.
        mol.set_adjacency()
        bonds = adjacency_to_bonds(mol.adjacency)

        # Initialise graph.
        mol_graph = nx.Graph()
        mol_graph.add_edges_from(bonds)

        # Initialise index list.
        new_index = []
        while len(new_index) < len(mol.atom_ids):
            # Set new start atom if disconnected graph or new mol.
            if len(new_index) > 0:
                # Find atoms not in index and set new start node.
                start = set(range(len(mol.atom_ids))).difference(set(new_index))
                start_atom = reference_mol.atom_ids[list(start)[0]]
            else:
                start_atom = reference_mol.atom_ids[start_node]
            # Find paths in molecule and add to new index list.
            molecule_paths = find_paths(start_node, mol, mol_graph)
            new_index += stack_paths(molecule_paths)
 
        # Reindex molecule.        
        mol.reindex_molecule(new_index)

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
        Recentered dihedral values.

    """ 
    return dihedral_vals.apply(lambda x: x - (x/abs(x))*180)


def calculate_dihedrals(molecules, dihedral_smarts, charge=0):
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
            rdkit_mol = molecule_to_rdkit(mol, charge=charge)
            # print(rdkit.MolToSmarts(rdkit_mol))

            # Find atom indexes for dihedral.
            query_mol = rdkit.MolFromSmarts(smarts)
            indexes = rdkit_mol.GetSubstructMatches(query=query_mol)
            
            # Save each set of indexes to dihedral dict.
            for j, ind in enumerate(indexes):
                dihedral_indexes[dihed + '_' + str(j)] = list(ind)
        
        # Calculate dihedral values.
        mol.set_parameters(dihedral_indexes)

    return molecules

def calculate_conformer_RMSD(molecules, reference_mol=None):
    """
    Calculate RMSD with lowest energy molecule.

    Parameters
    ----------
    molecules: `list` of :molLego:`Molecule`
        Molecules to calculate RMSD for.
    reference_mol: :molLego:`Molecule`
        Moelcule to match indexes to. [Default: None type]
        If `None` then defaults to lowest energy molecule in molecules list.
    
    Returns
    -------
    rmsd: `list`
        RMSD values for each molecule with the lowest energy molecule.
        
    """
    # Set reference molecule as lowest energy molecule if not set.
    if reference_mol is None:
        lowestE = molecules[0].escf
        reference_mol = molecule_to_rdkit(molecules[0], charge=molecules[0].charge)

        # Search for lowest energy molecule.
        for mol in molecules:
            if mol.escf < lowestE:
                reference_mol = molecule_to_rdkit(mol, charge=mol.charge)
    else:
        reference_mol = molecule_to_rdkit(reference_mol, charge=reference_mol.charge)

    # Calculate RMSD with lowest E conformer for each conformer.
    rmsd = []
    for mol in molecules:
        # Convert molecule to rdkit mol.
        rdkit_mol = molecule_to_rdkit(mol, charge=mol.charge)
        try:
            rmsd.append(AlignMol(rdkit_mol, reference_mol))
        except:
            rmsd.append(np.nan)
    
    return rmsd

def calculate_dihedral_deviation(conf_diheds, reference_diheds):
    """
    Calculate dihedral deviation using Manhatten Distance.
    
    Parameters
    ----------
    conf_diheds : :pandas:`DataFrame`
        Geometric data for starting conformation.
    reference_diheds : :pandas:`DataFrame`
        Geometric data for end conformation.

    Returns
    -------
    `float`
        Dihedral deviation between the two conformers.

    """
    num_dihedrals = len(conf_diheds)

    # Calculate recentred dihedrals.
    reference_diheds_rc = recentre_dihedrals(reference_diheds)
    conf_diheds_rc = recentre_dihedrals(conf_diheds)
    
    # Calculate normalised deviation for both dihedral sets.
    diff = np.abs(conf_diheds - reference_diheds)/180.
    diff_rc = np.abs(conf_diheds_rc - reference_diheds_rc)/180.

    # Calculate deviation using smallest differences.
    diff_sum = np.minimum(diff, diff_rc).sum()
    
    return diff_sum/num_dihedrals

def calculate_dihedral_deviation_l2(conf_diheds, reference_diheds):
    """
    Calculate dihedral deviation using Euclidean Distance.
    
    Parameters
    ----------
    conf_diheds : :pandas:`DataFrame`
        Geometric data for starting conformation.
    reference_diheds : :pandas:`DataFrame`
        Geometric data for end conformation.

    Returns
    -------
    `float`
        Dihedral deviation between the two conformers.

    """
    num_dihedrals = len(conf_diheds)

    # Calculate recentred dihedrals.
    reference_diheds_rc = recentre_dihedrals(reference_diheds)
    conf_diheds_rc = recentre_dihedrals(conf_diheds)
    
    # Calculate normalised deviation for both dihedral sets.
    diff = ((conf_diheds - reference_diheds)/180.)**2
    diff_rc = ((conf_diheds_rc - reference_diheds_rc)/180.)**2

    # Calculate  using smallest differences.
    sqrt_diff_sum = np.sqrt(np.minimum(diff, diff_rc).sum())
    
    return sqrt_diff_sum/num_dihedrals

def push_geom_xyz(output_file, molecule):
    """
    Output molecule to an .xyz file.

    Parameters
    ----------
    output_file : :class:`str`
        The name of the output xyz file.
    molecule : :Molecule:
        The molecule to be output.
    
    """
    # Open output file, print header lines then atom indexes and cartesian coordinates to file
    with open(output_file + '.xyz', 'w+') as out_file:
        print(molecule.atom_number, file=out_file)
        print('Structure of {}'.format(output_file.split('.')[0]), file=out_file)
        for atom_ind, atom in enumerate(molecule.atom_ids):
            print('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(atom, molecule.geometry[atom_ind]), file=out_file)

def parameter_dict_to_ids(parameter_dict):
    """
    Create geometric parameter ids from dict of parameters.

    Parameters
    ----------
    parameter_dict : :class:`dict`
        Key is atom types of geometric parameter.
        Value is atom index of geometric parameter.
    
    Returns
    -------
    parameter_ids : :class:`list` of `str`
        The parameter ids in the form atom_type atom_ind - ...
        for each atom involved in the paramter.
        Example: P0-S1
    
    """
    parameter_ids = []
    for param_type, param_ids in parameter_dict.items():
        param = [param_type[i] + str(param_ids[i]) 
                 for i in range(len(param_ids))]
        parameter_ids.append('-'.join(param))
    
    return parameter_ids
