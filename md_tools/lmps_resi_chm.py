"""Module to create LAMMPS input files from CHARMM FF (par/top/str)."""
import sys
import numpy as np
import itertools

from phdtbtk.md_tools.chm_resi import Resi, Atom
from phdtbtk.md_tools.parse_pdb import PDB

def parse_resi(current_line, in_file, residues, current_resi, resi_list):
    """
    Parse residue information from CHARMM toplogy file and add to dict.

    Parameters
    ----------
    current_line : :class:`str`
        Current line in file.

    in_file : iter object
        Lines of file.

    residues : :class:`dict`
        Key is the residue name.
        Value is the topology file lines 
        containing all residue information.

    current_resi : :class:`str`
        Reisde name for residue to be parsed.

    resi_list : :class:`list` of `str`
        List of target residues to parse. 

    Returns
    -------
    residues : :class:`dict`
        Updated residues dict with current resi as new entry.

    """
    current_line = next(in_file)
    # while 'RESI' or 'PRES' not in current_line:
    while 'RES' not in current_line:
        # Check for EOF.
        if 'END' in current_line:
            break
        # Append information for current residue.
        residues[current_resi].append(current_line.strip())
        current_line = next(in_file)
    
    # Parse next residue if required.
    for resi in resi_list:
        if (f"RESI {resi.upper()} " in current_line or 
            f"PRES {resi.upper()} " in current_line):
            residues = parse_resi(current_line, in_file, 
                                  residues, resi, resi_list)
    return residues

def parse_topology(top_files, resi_list):
    """
    Parse CHARMM topology and residues from file.
    
    Parameters
    ----------
    top_files : :class:`list` of `str`
        Topology files to parse.

    resi_list : :class:`list` of `str`
        List of target residues to parse. 
    
    Returns
    -------
    mass : :class:`dict`
        Key is atom types.
        Value is mass.

    residues : :class:`dict`
        Key is residue name.
        Value is `list` of residue lines from topology file.

    """
    # Initialise variables.
    mass = {}
    residues = {x: [] for x in resi_list}

    for topology in top_files:
        with open(topology, "r+") as in_file:
            for el in in_file:
                # Create dict of atom types.
                if 'MASS' in el:
                    mass[el.split()[2]] = float(el.split()[3])
                
                # Parse residue information.
                for resi in resi_list:
                    if (f"RESI {resi.upper()} " in el or 
                        f"PRES {resi.upper()} " in el):
                        residues = parse_resi(el, in_file, residues, resi, resi_list)
    return mass, residues

def bonds_from_line(input_line):
    """Convert input line to list of bonds."""
    atoms = input_line.split()
    return [[atoms[i], atoms[i+1]] for i in range(1, len(atoms), 2)]

def process_resi(resi_name, resi_info, mass):
    """
    Process residue information to Resi.

    Parameters
    ----------
    resi_name : :class:`str`
        Name of residue.
    
    resi_info : :class:`list` if `str`
        Residue information lines from CHARMM topology file.
    
    mass : :class:`dict`
        Key is atom types.
        Value is mass.

    Returns
    -------
    `Resi`
        Resi object of residue.

    """
    atoms, bonds = [], []
    angles = []
    dihedrals = []
    impropers = []
    
    bond_flags = ['BOND', 'DOUB', 'TRIP', 'AROM']
    angle_flags = ['ANGL', 'THET']
    improper_flags = ['IMPH', 'IMPR']
    
    for line in resi_info:
        if 'ATOM' in line:
            atom_name, atom_type = line.split()[1:3]
            atom_charge = float(line.split()[3])
            atom_mass = mass[atom_type]
            atoms.append(Atom(atom_name, atom_type, atom_charge, atom_mass))
        elif any([x in line for x in bond_flags]):
            bond_list = bonds_from_line(line)
            [bonds.append(x) for x in bond_list]
        elif any([x in line for x in angle_flags]):
            angles.append(line.split()[1:4])
        elif 'DIHE' in line:
            dihedrals.append(line.split()[1:5])
        elif any([x in line for x in improper_flags]):
            impropers.append(line.split()[1:5])
    # Set residue.
    return Resi(resi_name, atoms, angles, 
                bonds, dihedrals, impropers)

### PARAMETER FILE PARSING ###
def order_parameter(parameter):
    """Order parameter to put alphabetical atom type first."""
    # Convert to list if single str.
    if isinstance(parameter, str):
        parameter = parameter.split()
        back_to_str = True
    else: 
        back_to_str = False

    # Order parameter.
    outer = [parameter[0], parameter[-1]]
    if outer != sorted(outer):
        parameter.reverse()

    # Return parameter in same form as input type.
    if back_to_str:
        return ' '.join(parameter)
    else:
        return parameter

def params_to_dict(param_list, num_atoms):
    """
    Convert CHARMM parameter str lines to dict.

    Parameters
    ----------
    param_list : :class:`list` of `str`
        List of parameters containing atom types
        and parameters as unprocessed string.
    
    num_atoms : :class:`int`
        Number of atoms defining the parameter.
        E.g. 2 for bond, 3 for angle, 
             4 for dihedral/improper.
    
    Returns
    -------
    parameters : :class:`dict`
        Key is list of the atom types.
        Value is a list of parameter values. 

    """
    parameters = {}
    for param_line in param_list:
        param_line = param_line.split()
        try:
            atom_types = ' '.join(order_parameter(param_line[0:num_atoms]))
            # Save coefficients to atom type key.
            if atom_types not in parameters:
                parameters[atom_types] = [float(x) for x in 
                                          param_line[num_atoms:]]
            else:
                # Handle if multiple parameters for dihedrals with different multiplicity.
                if isinstance(parameters[atom_types][0], float):
                    parameters[atom_types] = [parameters[atom_types]]
                parameters[atom_types].append([float(x) for x in 
                                               param_line[num_atoms:]])
        except:
            print('Line not processed:\n', param_line)
    return parameters

def parse_parameters(prm_files):
    """
    Parse parameters from CHARMM parameter (.prm/str) files.

    Parameters
    ----------
    prm_files : :class:`list` of `str`
        Parameter files to parse.

    """
    # Initialise variables.
    param_flags = ['BONDS', 'ANGLES', 'DIHEDRALS', 'IMPROPERS', 
                   'NONBONDED', 'NBFIX']
    params = {x : [] for x in param_flags}
    
    for parameters in prm_files:
        param_type = None
        with open(parameters, "r+") as in_file:
            for el in in_file:
                
                # Switch parameter type being parsed on new section.
                for p_flag in param_flags:
                    if p_flag in el:
                        param_type = p_flag
                        el = next(in_file)
                
                # Parse atom types and parameters from line.
                if el.strip() and (param_type is not None):
                    if el[0] != '!':
                        params[param_type].append(el.split('!')[0])
    
    return params

def nonbonded_to_lammps(nonbond_chm):
    """
    Process CHARMM LJ parameters to LAMMPS LJ parameters.

    CHARMM FF has nonbonded parameters as -eps
    (by convention as E at min is -eps) and 
    r_min/2. LAMMPS requires conversion to eps and
    sigma (= r_min/2 * 2^(5/6)).

    Parameters
    ----------
    nonbond_chm : :class:`list` of `float`
        CHARMM coefficients for each parameter.
    
    Returns
    -------
    nonbond_lmps : :class:`list` of `float`
        LAMMPS coefficients for each parameter.

    """
    # Initialise parameters.
    nonbond_lmps = []
    
    # Remove ignored parameters. Change -eps to eps and rmin/2 to sigma.
    for nb in nonbond_chm:
        nb_new = []
        # Range allows handling of possible 1,4 parameters.
        for i in range(1, len(nb), 3):
            nb_new.append(np.abs(nb[i]))
            nb_new.append(nb[i+1]*(2**(5/6)))
        nonbond_lmps.append(nb_new)
    
    return nonbond_lmps

def get_param_str(resi, atom_inds):
    """Convert atom indexes (1 index) to atom type parameter string."""
    atom_inds = [(x - 1) for x in atom_inds]
    atom_types = order_parameter(resi.get_atom_type(atom_inds))
    return ' '.join(atom_types)

def check_duplicates(resi, coeffs, types, indexes):
    """Check for and add atom types with multiple parameters."""
    # Find instances with multiple coefficients.        
    duplicates = []
    for i, coeff in enumerate(coeffs):
        if isinstance(coeff[0], list):
            duplicates.append(types[i])

    # Remove mutli param instances from original lists and add repeats.
    for dup in duplicates:
        coeff = coeffs.pop(types.index(dup))
        coeffs += coeff
        types.remove(dup)
        types += [dup for x in range(len(coeff))]
    
        # Convert indexes to atom types.
        dup_indexes = []
        for dihed in indexes:

            # Find multi param atom types.
            if get_param_str(resi, dihed) == dup:
                dup_indexes.append([dihed]*len(coeff))

        # Remove original repeated index and add repeats.
        [indexes.remove(di[0]) for di in dup_indexes]
        indexes += [di for di in itertools.chain(*dup_indexes)]

    return coeffs, types, indexes

def set_coeffs(resi, params, parameters):
    """
    Set coefficient information for LAMMPS data file.

    Parameters
    ----------
    resi : :class:`Resi`
        Residue.

    params : :class:`str`
        Parameter attribute of resi.
        Must be one of 'bonds', 'angles',
        'dihedrals' or 'impropers'.

    parameters : :class:`dict`
        Where Key is a list of the atom types 
        and Value is a list of parameter values. 

    Returns
    -------
    types : :class:`list` of `str`
        Atom types for atoms defining geometric parameter.
    
    coeffs : :class:`list` of `float`
        Coefficients defining parameter.

    """
    # Get atom type and convert to string to get parameter values.
    types = [resi.get_atom_type(x) for x in getattr(resi, params)]

    # Order to match and change to parameter key string.
    types = list(dict.fromkeys([' '.join(order_parameter(x)) for x in types]))
    coeffs = [parameters[x] for x in types]
    
    return types, coeffs

def set_indexes(resi, params):
    """
    Set topology information for LAMMPS data file.

    Parameters
    ----------
    resi : :class:`Resi`
        Residue.
    params : :class:`str`
        Parameter attribute of resi.
        Must be one of 'bonds', 'angles',
        'dihedrals' or 'impropers'.

    Returns
    -------    
    indexes : :class:`list` of `str`
        Indexes of atoms defining geometric parameter.

    """
    # Get indexes defining parameter topology.
    indexes = []
    atom_names = [atom.name for atom in resi.atoms]
    for param in getattr(resi, params):
        indexes.append([atom_names.index(x)+1 for x in param])

    return indexes

def format_coefficients(coeff_set, param_style, param_types):
    """
    Write formatted output string for parameter coefficients.

    Parameters
    ----------
    coeff_set : :class:`list` of `float`
        List of coefficients for each parameter.
    
    param_style : :class:`str`
        Parameter type and style to specify formatting.
        If format type not specified then defaults to float print out
        with no specific formatting.
    
    param_types : :class:`list` of `str`
        Atom types defining parameter.

    Returns
    -------
    output : :class:`list` of `str`
        Formatted output line of coefficients for each parameter.

    """
    # Map of parameter style to format.
    formats = {'harmonic': ['16.8f', '16.8f'],
               'angle_charmm' : ['16.8f', '16.8f', '16.8f', '16.8f'],
               'dihed_charmm': ['16.8f', '8.0f', '8.0f', '16.8f'],
               'impr_harmonic': ['16.8f', '8.0f', '8.0f']}

    # Write output string for coefficients.
    output = []    
    for i, coeff in enumerate(coeff_set):
        output_str = f"{i+1:6}"
        
        # Use format if specified.
        if param_style in formats:
            # Add dummy coefficients if too short (CHARMM angles).
            while len(coeff) < len(formats[param_style]):
                coeff.append(0.0)
            if param_style == 'impr_harmonic':
                coeff[1] = 1.0
            for j, x in enumerate(coeff):
                output_str += (f"{x: {formats[param_style][j]}}")
        
        # Use general format if style not defined.
        else:
            for x in coeff:
                output_str += (f"{x: 16.8f}")
        
        # Add atom types as comment.
        output_str += f" # {param_types[i]}"
        output.append(output_str)

    return output

def resi_to_lammps(resi, resi_geometry, parameters, box=[[0.0, 10.0]],lammps_output=None):
    """
    Write LAMMPS data file for residue.

    Parameters
    ----------
    resi : :class:`Resi`
        Residue.
    
    resi_geometry : :class:`numpy ndarray`
        A ``(N, 3)`` array of x, y, z coordinates for each atom.
            Where N is the number of atoms in the resi.
    
    parameters : :class:`dict`
        Where Key is a list of the atom types 
        and Value is a list of parameter values. 
    
    box : :class:`list` of `list`
        x, y, z upper and lower limits of box dimenisons.
        Must be of length 3 or 1. If length is one then sets a cubic 
        box of the single dimension provided.
    
    lammps_output :class:`str`
        Name of LAMMPS data file.
        [Default: None type] 
        If None sets to residue name + 'single_resi.data'.

    """
    # Properties of atom types.
    types = list(dict.fromkeys([atom.atype for atom in resi.atoms]))
    masses = [x[0] for x in resi.get_atom_type_prop(types, prop='mass').values()]
   
    # Retrieve and convert non bonded parameters.
    pair_coeffs = (parameters['nonbonded'][x] for x in types)
    pair_coeffs = nonbonded_to_lammps(pair_coeffs)

    # Atomic properties.
    atom_types = [types.index(atom.atype)+1 for atom in resi.atoms]
    charges = [atom.charge for atom in resi.atoms]

    # Bond properties.
    bond_types, bond_coeffs = set_coeffs(resi, 'bonds', parameters['bonds'])
    bond_indexes = set_indexes(resi, 'bonds')
    
    # Angle properties.
    angle_types, angle_coeffs = set_coeffs(resi, 'angles', parameters['angles'])
    angle_indexes = set_indexes(resi, 'angles')
    
    # Dihedral properties.
    dihedral_types, dihedral_coeffs = set_coeffs(resi, 'dihedrals', parameters['dihedrals'])
    dihedral_indexes = set_indexes(resi, 'dihedrals')
    # Check for multiple parameters for dihedral multiplicities.
    dihedral_coeffs, dihedral_types, dihedral_indexes = check_duplicates(
        resi,
        dihedral_coeffs, 
        dihedral_types,
        dihedral_indexes
        )
 
    # Set topology as list for printing.
    topologies = list(zip([bond_indexes, angle_indexes, dihedral_indexes],
                     [bond_types, angle_types, dihedral_types]))
    coeff_input = {'nonbond': [pair_coeffs, types],
                   'bond': [bond_coeffs, bond_types], 
                   'angle': [angle_coeffs, angle_types], 
                   'dihedral': [dihedral_coeffs, dihedral_types]}

    # Set impropers if any in residue.
    if len(resi.impropers) > 0:
        improper_types, improper_coeffs = set_coeffs(resi, 'impropers', parameters['impropers'])
        improper_indexes = set_indexes(resi, 'impropers')
        topologies.append([improper_indexes, improper_types])
        coeff_input['improper'] = [improper_coeffs, improper_types]
    
    ### WRITE LAMMPS DATA FILE FOR RESIDUE ###
    if lammps_output is None:
        lammps_output = (resi.name).lower() + "_single_resi.data"
    with open(lammps_output, 'w+') as outfile:

        print(f"# {resi.name} single residue data file with CHARMM parameters.", file=outfile)
        print("", file=outfile)

        # System definition section.
        print(f"{len(resi.atoms)} atoms", file=outfile)
        print(f"{len(bond_indexes)} bonds", file=outfile)
        print(f"{len(angle_indexes)} angles", file=outfile)
        print(f"{len(dihedral_indexes)} dihedrals", file=outfile)
        print(f"{len(resi.impropers)} impropers", file=outfile)
        print("", file=outfile)

        # Type definitions.
        print(f"{len(types)} atom types", file=outfile)
        print(f"{len(bond_types)} bond types", file=outfile)
        print(f"{len(angle_types)} angle types", file=outfile)
        print(f"{len(dihedral_types)} dihedral types", file=outfile)
        if len(resi.impropers) > 0:
            print(f"{len(improper_types)} improper types", file=outfile)
        print("", file=outfile)

        # Box dimensions. Currently just sets to 0, 0, 0 ahead of packing.
        if len(box) == 1:
            box = [box[0]]*3
        for i, ax in enumerate(['x', 'y', 'z']):
            print(f"{box[i][0]:8.6f} {box[i][1]:8.6f} {ax}lo {ax}hi", file=outfile)
        print("", file=outfile)

        # Print Masses.
        print('Masses', file=outfile)
        print("", file=outfile)
        for i, mass in enumerate(masses):
            print(f"{i+1: 6}{mass: 10.6} # {types[i]}", file=outfile)
        print("", file=outfile)

        # Print coefficient sections.
        coeff_headers = ['Pair Coeffs', 'Bond Coeffs', 'Angle Coeffs', 
                         'Dihedral Coeffs', 'Improper Coeffs']
        coeff_styles = ['non_bonded', 'harmonic', 'angle_charmm', 'dihed_charmm', 
                        'impr_harmonic']
        for i, coeff_set in enumerate(coeff_input.values()):
            print(f"{coeff_headers[i]}", file=outfile)
            print("", file=outfile)
            output = format_coefficients(coeff_set[0], coeff_styles[i], coeff_set[1])
            for el in output:
                print(el, file=outfile)
            print("", file=outfile)

        # Write atoms sections.
        mol_num = 1
        print('Atoms', file=outfile)
        print("", file=outfile)
        for i, atom in enumerate(resi_geometry):
            print(
                f"{i+1:6}{mol_num:6}{atom_types[i]:6}{charges[i]:12.6f}"
                f"{atom[0]:12.6f}{atom[1]:12.6f}{atom[2]:12.6f}"
                f" # {types[atom_types[i]-1]}", 
                file=outfile)
        print("", file=outfile)

        # # Print Types and charges.
        # print('Types', file=outfile)
        # print("", file=outfile)
        # for i, atype in enumerate(atom_types):
        #     print(f"{i:6}{atype:6}", file=outfile)
        # print("", file=outfile)
        # print('Charges', file=outfile)
        # print("", file=outfile)
        # for i, charge in enumerate(charges):
        #     print(f"{i:6}{mol_num:6}{charge: 10.6f}", file=outfile)
        # print("", file=outfile)

        # Print topology sections.
        topology_headers = ['Bonds', 'Angles', 'Dihedrals', 'Impropers']
        for i, top_set in enumerate(topologies):
            print(f"{topology_headers[i]}",file=outfile)
            print("", file=outfile)
            ind_shft = 1
            for i, top in enumerate(top_set[0]):
                if top == top_set[0][i-1] and len(top_set[0])>1:
                    ind_shft += 1
                else:
                    ind_shft = 1

                top_str = get_param_str(resi, top)
                top_type = list(dict.fromkeys(top_set[1])).index(top_str) + ind_shft
                output_str = f"{i+1:6}{top_type:6}"
                for atom in top_set[0][i]:
                    output_str += f"{atom:8}"
                print(output_str, file=outfile)
            print("", file=outfile)

if __name__ == "__main__":

    ### PARSE CHARMM FF ###
    # Parameter and topology files.
    nagent = 'a232'
    na_str_path = '/Users/smf115/Box/naFF/chargesRESP/resp_str/'
    # na_str_path = '/Users/sophiemay/Box/naFF/chargesRESP/resp_str/'
    # na_str_path = '../../naFF/chargesRESP/resp_str/'
    prm_files = ['par_all36_cgenff.prm', na_str_path+nagent+'_penalty.str']
    top_files = ['top_all36_cgenff.rtf', na_str_path+nagent+'_penalty.str']
    # top_files = [na_str_path+nagent+'_penalty.str']
    # resi_list = ['ETOH']
    resi_list = ['A23']

    # Parse topology files and create Residues.
    mass, residues = parse_topology(top_files, resi_list)
    resi_list = []
    for resi_name, resi in residues.items():
        resi_list.append(process_resi(resi_name, resi, mass))

    # Parse parameter files and create parameter dicts.
    params = parse_parameters(prm_files)
    parameters = {}
    parameters['bonds'] = params_to_dict(params['BONDS'], 2)
    parameters['angles'] = params_to_dict(params['ANGLES'], 3)
    parameters['dihedrals'] = params_to_dict(params['DIHEDRALS'], 4)
    parameters['impropers'] = params_to_dict(params['IMPROPERS'], 4)
    parameters['nonbonded'] = params_to_dict(params['NONBONDED'], 1)
    
    # Read in geometry from PDB.
    pdb_file = nagent+'_init.pdb'
    resi_pdb = PDB(pdb_file)

    ### WRITE LAMMPS INPUT ###
    resi_to_lammps(resi_list[0], resi_pdb.geometry, parameters)







