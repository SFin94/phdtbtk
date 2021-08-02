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

class CharmmCRD():
    """
    Represents a CHARMM CRD file.
    
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

    resi : :class:`str`
        Name of residue of molecule.

    """
    
    def __init__(self, output_file):
        """
        Initialise molecule from CRD file.

        Parameters
        ----------
        output_file : :class:`str`
            File path for output CRD file.

        """
        self.file_name = output_file
        
        # Parse contents of file.
        with open(self.file_name, 'r+') as infile:
            for el in infile:
                # Skip header lines.
                if "*" not in el:
                    # Parse geometry.
                    self.atom_number = int(el.split()[0])
                    molecule_info = [next(infile).strip() for x in range(self.atom_number)]
        
        # Process attributes from molecule information.
        self.resi = molecule_info[0].split()[2]
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
                float(line.split()[i+4])
                for i in range(3)
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
        atoms = [x.split()[3] for x in molecule_info]
        atom_ids = [re.search(r'[\D_]', x).group(0) for x in atoms]

        return atoms, atom_ids

    
