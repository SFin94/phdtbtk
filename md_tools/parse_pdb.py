"""Parse geom from crd and pdb files."""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlin
import re

import molLego as ml
import phdtbtk

class PDB():
    """
    Represents a PDB file.
    
    Attributes
    ----------
    atom_ids : :class:`list` of `str`
        The atomic symbols of the atoms in the molecule.

    atoms : :class:`list` of `str`
        The atom names of the atoms in the molecule.

    atom_number : :class:`int`
        The number of atoms in the molecule.
 
    charge : :class:`int`
        The charge of the molecule.
        
    file_name : :class:`str`
        The path to the parent log file.
    
    geometry : :class:`numpy ndarray`
        A ``(N, 3)`` array of x, y, z coordinates for each atom.
        Where N is the number of atoms in the molecule.
        
    resi : :class:`str`
        Name of residue of molecule.

    """
    
    def __init__(self, output_file):
        """
        Initialise molecule from PDB file.

        Parameters
        ----------
        output_file : :class:`str`
            File path for output PDB file.

        NEEDS ADAPTING TO MULTIPLE MOLECULE/RESI
        """
        self.file_name = output_file
        molecule_info = []
        # Parse contents of file.
        with open(self.file_name, 'r+') as infile:
            for el in infile:
                # Parse molecule information.
                while any(["ATOM" in el, "HETATM" in el]):
                    molecule_info.append(el)
                    el = next(infile)
        
        self.atom_number = len(molecule_info)
        
        # Process attributes from molecule information.
        self.resi = molecule_info[0].split()[3]
        self.geometry = self.process_geometry(molecule_info)
        self.atoms, self.atom_ids = self.process_atoms(molecule_info)

    def process_geometry(self, molecule_info):
        """
        Process the cartesian coordinate geometry from molecule info.

        Parameters
        ----------
        molecule_info : :class:`list` of `str`
            Nested list containing unprocessed molecule
            information from crd file.

        Returns
        -------
        :class:`numpy.ndarray`
            A ``(N, 3)`` array of x, y, z positions for each atom.
            Where N is the number of atoms in the molecule.

        """
        # Initalise variables
        atom_coords = []

        # Pull coordinates from molecule info.
        for line in molecule_info:
            xyz = np.asarray([
                float(x) for x in line[30:55].split()
            ])
            atom_coords.append(xyz)

        return np.asarray(atom_coords)


    def process_atoms(self, molecule_info):
        """
        Process atom information from molecule info.

        Parameters
        ----------
        molecule_info : :class:`list` of `str`
            Nested list containing unprocessed molecule
            information from crd file.

        Returns
        -------
        atom_ids : :class:`list` of `str`
            The atomic symbols of the atoms in the molecule.

        atoms : :class:`list` of `str`
            Atom identifier (atom id and index) from file.

        """
        # Set atoms from molecule information.
        atoms = [x[12:].split()[0] for x in molecule_info]
        atom_ids = [re.search(r'[\D_]', x).group(0) for x in atoms]

        return atoms, atom_ids

    
