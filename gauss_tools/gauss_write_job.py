"""Module containing workflow and functions to write or edit Gaussian com file."""

import sys
import argparse
import numpy as np
import pandas as pd

from phdtbtk.gauss_tools.gauss_com import GaussianCom
import molLego as ml


"""Has functions for parsing gometry, charge, multiplicty from different sources.
The main routine sets a GaussianCom object and writes the file."""

# Change path to location of presets file here
presets_path = '/Volumes/home/bin/.presets'

def parse_com_file(initial_com_file, section=2):
    """
    Parse charge and multiplicity and geometry from initial com file.
    
    Parameters
    ----------
    initial_com_file : :class:`str`
        The file path/name of input .com file.
    section : :class:`list` of `int`
        Key of target block to pull of .com file.
        Each 'target block' is the .com input section 
        seperated by empty lines.
            2 - Pull charge/multiplicty and geometry section [default].
            3 - Pull modredundant input.
            Possilibity for other pre done input (e.g. pseudopotential)
            but must know which section is required.

    Returns
    -------
    section_output : :class:`list` of `str`
        The lines from the targeted block of the input .com file.
    
    """
    # Initialise variables
    new_section_flag = ''
    section_count = 0
    section_output = []

    # Parse file and read through till target section reached
    with open(initial_com_file, 'r') as input:
        for line in input:
            if line.strip() == new_section_flag:
                section_count += 1
            
            # Pull all lines of target section
            elif section_count == section:
                section_output.append(line.strip())
    
    return section_output

def parse_comp_presets(preset):
    """
    Set computational resources from line in presets file.

    Parameters
    ----------
    preset : :class:`int`
        Index of desired line (1 index) in presets file 
        to set resources from.

    Returns
    -------
    nproc : :class:`int`
        Number of prcoessors required for calculation.
    mem_MB : :class:`int`
        Amount of memory (MB) required for calculation.

    """
    # Parses in presets from defined file and sets computational variables
    try:
        with open(presets_path, 'r') as presets:
            preset_num = 0
            for el in presets:
                if el[0] != '#':
                    preset_num += 1
                if preset_num == (preset+1):
                    nproc = int(el.split(';')[2])
                    mem_MB = int((el.split(';')[3])[:-2]) - nproc*100
    except IOError:
        print("Couldn't locate the presets file in " + presets_path, sys.stderr)
    
    return nproc, mem_MB

def geom_from_log(input_file, geom_step=None):
    """Extract geometry and atom ids from .log file."""
    molecule = ml.Molecule(input_file, parser=ml.GaussianLog)
    if geom_step is not None:
        trajectory_step = molecule.parser.pull_trajectory(calculation_steps=geom_step)[geom_step]
        molecule.geometry = trajectory_step['geom']
    return molecule.geometry, molecule.atom_ids

def geom_from_com(input_file, geom_step=None):
    """Extract geometry as coordinates and atom ids from .com file."""
    molecule_spec = parse_com_file(input_file)[1:]
    atom_ids = []
    atom_coords = []
    for line in molecule_spec:
        atom_ids.append(line.split()[0])
        xyz = np.asarray([
            float(line.split()[i+1]) 
            for i in range(3)
        ])
        atom_coords.append(xyz)

    return np.asarray(atom_coords), atom_ids

def cm_from_log(input_file):
    """Extract charge/multplicity from .log file."""
    molecule = ml.Molecule(input_file, parser=ml.GaussianLog)
    multiplicity = molecule.parser.pull_multiplicity()
    return molecule.charge, multiplicity
    
def cm_from_com(input_file):
    """Extract charge/multplicity from .com file."""
    molecule_spec = parse_com_file(input_file)[0]
    charge, multiplicity = [int(i) 
        for i in molecule_spec.split()]
    return charge, multiplicity

def get_molecule(input_file, pull_geometry=True, 
                 pull_cm=False, geom_step=None):
    """
    Retrieve molecule information from specified sources.

    Parameters
    ----------
    input_file : :class:`str`
        File containing molecule informaton to retrieve.
    pull_geometry : :class:`bool`
        If ``True`` retrives geometry and atom ids from input file.
    pull_cm : :class:`bool`
        If ``True`` retrives charge/multiplicty from input file.

    Returns
    -------
    mol_spec : :class:`list`
        List of molecule information depending on target.
        Would contain: geometry, atom ids, charge/multiplicity 

    """
    # Use input file type to map to function for setting molecule spec
    mol_spec = []
    geometry_functions = {'log': geom_from_log, 'com': geom_from_com}
    cm_functions = {'log': cm_from_log, 'com': cm_from_com}
    
    # Pull geometry and atom ids for molecule depending on file type
    if pull_geometry:
        input_file_type = input_file.split('.')[-1]
        mol_spec += geometry_functions[input_file_type](input_file, geom_step)

    # Pull charge and multiplicity from input file or raw entry
    if pull_cm:
        if type(input_file) == str:
            input_file_type = input_file.split('.')[-1]
            mol_spec += cm_functions[input_file_type](input_file)
        else:
            try:
                mol_spec += input_file
            except:
                print('Error setting charge and multiplicity, provide file source or list of values')
    return mol_spec

def test_input(molecule_input, cm_input):

    test_values = [i is None for i in [molecule_input, cm_input]]
    if all(test_values):
        return 'allchk', 'allchk'
    elif test_values == [True, False]:
        return 'chk', cm_input
    elif test_values == [False, True]:
        if molecule_input == 'chk':
            raise Exception('chk selected for geometry input but no charge/multiplicity provided.')
        else:
            return molecule_input, molecule_input
    else:
        return molecule_input, cm_input


def push_com(output_file, job_type='fopt', preset=None, 
             nproc=20, mem=62000, method='M062X', 
             basis_set='6-311++G(d,p)', smd=None, 
             modred_input=None, molecule_input=None, 
             geom_step=None, cm_input=None):
    """
    Create a GaussianCom instance and write a new .com file.
    
    Parameters
    ----------
    output_file : :class:`str`
        Name/path of output .com file.
    job_type : :class:`str`
        Gaussian calculation type.
        [Default: 'fopt']
    preset : :class:`int`
        Line number of preset to use for computational resources.
    nproc : :class:`int`
        Number of prcoessors required for calculation.
        [Default: 20]
    mem : :class:`int`
        Amount of memory required for calculation.
        [Default: 62000]
    method : :class:`str`
        Calculation method to use.
        (functional, etc.) [Default: 'M062X']
    basis_set : :class:`str`
        Basis set to use for the calculation. 
        [Default: '6-311++G(d,p)']
    smd : :class:`bool`
        ``True`` if SMD implicit solvent model for water required.
        Otherwise gas phase calculation. [Default: False]
    modred_input : :class:`str`
        Modredundant input lines. NoneType if
        no Modredundant input required [default: None].
    molecule_input : :class:`str`
        File/source for molecule geometry.
    cm_input : :class:`str`
        File/source for molecule charge/multiplicty.
    
    """
    # Set computatonal resources from preset file.
    if preset != None:
        nproc, mem = parse_comp_presets(preset)

    # Process cm/molecule input options to set them to appropriate values if either are None
    molecule_input, cm_input = test_input(molecule_input, cm_input)

    # If allchk then call com file with no molecule spec
    if 'allchk' in molecule_input:
        new_com_file = GaussianCom(output_file, job_type=job_type, 
                                   nproc=nproc, mem=mem, method=method,
                                   basis_set=basis_set, smd=smd, 
                                   modred_input=modred_input)
    else:
        # Set charge and multilpicty from file source or directly if given
        charge, multiplicity = get_molecule(cm_input, 
                                            pull_geometry=False, 
                                            pull_cm=True)   

        # If chk then call com file with no atom ids or geometry
        if 'chk' in molecule_input:
            new_com_file = GaussianCom(output_file, job_type=job_type, 
                                       nproc=nproc, mem=mem, 
                                       method=method, 
                                       basis_set=basis_set, smd=smd, 
                                       modred_input=modred_input, 
                                       charge=charge, 
                                       multiplicity=multiplicity)
        
        # Set geometry and atom ids from source file
        else:
            geometry, atom_ids = get_molecule(molecule_input, geom_step=geom_step)
            new_com_file = GaussianCom(output_file, job_type=job_type, 
                                       nproc=nproc, mem=mem, 
                                       method=method, 
                                       basis_set=basis_set, smd=smd, 
                                       modred_input=modred_input, 
                                       charge=charge, 
                                       multiplicity=multiplicity, 
                                       geometry=geometry, 
                                       atom_ids=atom_ids)

    # Write .com file
    new_com_file.write_com_file()


if __name__ == "__main__":

    '''Argparse usage and arguments'''

    usage = "usage: %(prog)s [output_file] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("output_file", type=str, 
                        help="The name of the .com file to be written [without .com]")
    parser.add_argument("-j", "--job", dest="job_type", 
                        type=str, default='fopt', 
                        choices=['freq', 'opt', 'reopt', 'fopt', 'scan', 'ts', 'own'],
                        help="Gaussian job type, currently available: "
                        "opt, freq, fopt [opt+freq], reopt (opt+freq "
                        "from chk), scan (opt(ModRedundant)), ts "
                        "(TS Opt), own (enter own arguments).")
    parser.add_argument("-m", "--method", dest="method", 
                        type=str, default='M062X',
                        help="The method to be used, give the correct "
                        "gaussian keyword.")
    parser.add_argument("-b", "--basis_set", dest="basis_set", 
                        type=str, default='6-311++G(d,p)',
                        help="The basis set to be used, give the correct "
                        "gaussian keyword.")
    parser.add_argument("-g", "--geom", dest="geom_input", 
                        nargs='*', default=['allchk'],
                        help="Input source file for molecule geometry "
                        "and atom ids (and charge/multiplicity if same "
                        "source) or chk/allchk. A specific optimisation "
                        "step can be provided if log file.")
    parser.add_argument("-c", "--cm", dest="cm_input", nargs='*', 
                        help="Input source for charge and multiplicity of molecule, either "
                        "file source or values in order of charge multiplicity (e.g. 0 1)")
    parser.add_argument("--mod", dest="modred_input", type=str,
                        help="ModRedundant input, each input line entered as a csv if multiple")
    parser.add_argument("--smd", dest="smd", action='store_true',
                        help="Flag whether to include SMD keyword for solvation or not, set to true for input")
    parser.add_argument("-p", dest="preset", nargs=1, type=int,
                        help="Preset flag to set required prcoessors and mem from presets file")
    parser.add_argument("--nproc", dest="nproc", type=int, default=20,
                        help="Number of processors")  
    parser.add_argument("--mem", dest="mem", type=int, default=62000,
                        help="Memory requirement of job (MB)")

    args = parser.parse_args()
    
    # Process geom input if optstep is given.
    if len(args.geom_input) == 2:
        geom_input = args.geom_input[0]
        geom_step = int(args.geom_input[1])
    else:
        geom_input = args.geom_input[0]
        geom_step = None

    # Process cm input for either raw input or file name.
    if args.cm_input is not None and len(args.cm_input) == 1:
        cm_input = args.cm_input[0]
    else:
        cm_input = args.cm_input

    # Convert modred input to list.
    if args.modred_input is not None:
        modred_input = args.modred_input.split(',')
    else:
        modred_input = None

    # Call method to create and write com file.
    push_com(args.output_file, job_type=args.job_type, 
             preset=args.preset, nproc=args.nproc, 
             mem=args.mem, method=args.method, 
             basis_set=args.basis_set, smd=args.smd, 
             modred_input=modred_input, 
             molecule_input=geom_input, 
             geom_step=geom_step, cm_input=cm_input)
