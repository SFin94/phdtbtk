"""Module containing Resi and Atom (Resi building block) classes."""
import sys
import networkx as nx

class Atom():
    """
    Represents an atom in a residue.

    Attributes
    ----------
    name : :class:`str`
        The identifier/name of the atom. e.g. C1.

    atype : :class:`str`
        The atom type in the FF of the atom.
    
    charge : :class:`int`
        The partial charge of the atom.

    mass : :class:`float`
        Atomic mass of the atom.

    """

    def __init__(self, name, atom_type, charge, mass):
        """
        Initialise atom.

        Parameters
        ----------
        atom_type : :class:`str`
            The atom type in the FF of the atom.

        charge : :class:`float`
            The charge of the atom.

        mass : :class:`float`
            The mass of the atom.

        name : :class:`str`
            The unique identifier/name of the atom. e.g. C1.

        """
        self.name = name
        self.atype = atom_type
        self.charge = charge
        self.mass = mass


# Currently doesn't handle lone pairs.
class Resi():
    """
    Represents a RESI from CGENFF.

    Attributes
    ----------
    atoms : :class:`list` of :Atom:
        The atoms in the residue.
    
    bonds : :class:`list` of `str`
        The bonds in the residue.

    angles : :class:`list` of `str`
        The angles in the residue.

    dihedrals : :class:`list` of `str`
        The dihedrals in the residue.

    impropers : :class:`list` of `str`
        The impropers in the residue.

    lonepairs : :class:`list` of `str`
        The lone pairs in the residue.
    
    name : :class:`str` 
        The name of the residue.

    """
    
    def __init__(self, resi_name, atoms, angles, 
                bonds, dihedrals, impropers):
        """
        Intiialise residue.

        Parameters
        ----------
        atoms : :class:`list` of :Atom:
            The atoms in the residue.

        angles : :class:`list` of `str`
            The angles in the residue.

        bonds :class:`list` of `str`
            The bonds in the residue.

        dihedrals : :class:`list` of `str`
            The dihedrals in the residue.

        impropers : :class:`list` of `str`
            The impropers in the residue.

        name : :class:`str`
            The name of the residue.
        
        """
        self.name = resi_name
        self.atoms = atoms
        self.angles = angles
        self.bonds = bonds
        self.dihedrals = dihedrals
        self.impropers = impropers
        
        # Autogenerate dihedrals or angles if empty.
        autogen = [not x for x in [self.angles, self.dihedrals]]
        if any(autogen):
            self.autogen_connections(autogen[0], autogen[1])

    def _check_not_linear(self, graph, path):
        for i in [[0, 1], [3, 2]]:
            if (len(graph[path[i[0]]]) == 1 and 
                len(graph[path[i[1]]]) == 2):
                return False
        return True

    def autogen_connections(self, angles=True, dihedrals=True):
        """
        Generate dihedral and/or angle lists from bonds.

        Parameters
        ----------
        angles : :class:`bool`
            If ``True`` sets angles.
        
        dihedrals : :class:`bool`
            If ``True`` sets dihedrals.

        """
        # Initialise graph from bonds and atoms.
        atom_list = [a.name for a in self.atoms]
        resi_g = nx.Graph()
        resi_g.add_nodes_from([a.name for a in self.atoms])
        resi_g.add_edges_from(self.bonds)
         
        # Find all angles and dihedrals as paths in molecule.
        visited = []
        for i, start_node in enumerate(resi_g):
            visited.append(start_node)
            for end_node in resi_g:
                if end_node not in visited:
                    paths = nx.all_simple_paths(resi_g, start_node, end_node, cutoff=4)
                    # Add paths to angle or dihedral list.
                    if paths:
                        for p in paths:
                            if angles and len(p) == 3:
                                self.angles.append(p)
                            elif dihedrals and len(p) == 4:
                                if self._check_not_linear(resi_g, p):
                                    self.dihedrals.append(p)
    
    def get_atom_type(self, atoms=None):
        """
        Convert atom names or indexes to atom types.

        Parameters
        ----------
        atoms : :class:`list` of `str` or `int`
            Names or indexes of atoms to retrieve atom types for.
            [Default: ``None``]. If ``None`` returns atom types
            for all atoms in residue.
            Can be single `str` or `int` for single atom.
        
        Returns
        -------
        atom_types : :class:`list` of `str`
            Atom types of atoms.
        
        """
        # Set to all atom names if none given.
        if atoms is None:
            atoms = [x for x in range(len(self.atoms))]
        elif isinstance(atoms, (str, int)):
            atoms = (atoms, )
        
        # Find corresponding atom type for each atom name.
        atom_types = []
        resi_atoms = [atom.name for atom in self.atoms]
        for atom in atoms:
            if isinstance(atom, str):
                atom_ind = resi_atoms.index(atom)
            else:
                atom_ind = atom
            atom_types.append(self.atoms[atom_ind].atype)    

        return atom_types

    def get_atom_type_prop(self, atom_types=None, prop='mass'):
        """
        Get atom names, masses and charges for an atom type.

        Parameters
        ----------
        atom_types : :class:`list` of `str`
            Atom types to get properties for.
            [Default: ``None``]. If ``None`` returns for 
            all atom types in residue.
            Can be single `str` for single atom.
        
        prop : :class:`str`
            Attribute of `Atom` to be returned.
            One of: 'name', 'mass', 'charge'.
        
        Returns
        -------
        property : :class:`dict`
            Key is atom type and Value is attribute of `Atom`
            of that atom type.
        
        """
        # Set to all atom names if none given.
        if atom_types is None:
            atom_types = [atom.atype for atom in self.atoms]
        elif isinstance(atom_types, str):
            atom_types = (atom_types, )
        
        # Find corresponding atom type for each atom name.
        atom_prop = {x: [] for x in atom_types}
        for atom in self.atoms:
            if atom.atype in atom_types:
                atom_prop[atom.atype].append(getattr(atom, prop))
        return atom_prop
