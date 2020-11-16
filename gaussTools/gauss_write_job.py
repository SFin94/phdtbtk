import sys
import argparse
import numpy as np
import pandas as pd

import molLego as ml


# Change path to location of presets file here
presets_path = '/Volumes/home/bin/.presets'

class GaussianCom():
    """
    Represents a Gaussian com file.

    Attributes
    ----------
    output_file : :class:`str`
        Filepath/name for output file.
    nproc : :class:`int`
        Number of prcoessors required for calculation.
    mem : :class:`int`
        Amount of memory required for calculation.
    job_title: :class:`str`
        The title line for the job.
    molecule_spec: :class:`list of str`
        The molecule specification input lines (charge and multiplicity, x y z coordinates).
    job_spec : :class:`str`
        The job specification line containing all Gaussian commmands for the calculation.
    modred_input: :class:`list of str`
        The modredundant input lines.
    allchk : :class:`bool`
        If ``True`` no molecule specification or job title required.
    chk : :class:`bool`
        If ``True``, molecule specification will contain only charge and multiplicity.

    """

    def __init__(self, output_file, job_type='fopt', nproc=20, mem=62000, method='M062X', basis_set='6-311++G(d,p)', smd=None, modred_input=None, geometry=None, atom_ids=None, charge=None, multiplicity=None):
        """
        Initialise from input parameters.

        Parameters
        ----------
        output_file : `str`
            The file name/path of the output com file.
        job_type : `str`
            The type of calculation [one of: freq, fopt, opt, ts, reopt, scan; default: fopt].
        nproc : `int`
            Number of prcoessors required for calculation [default: 20].
        mem : `int`
            Amount of memory required for calculation [default: 62000 MB].
        method : `str`
            The method of the calculation (functional, etc.) [default: 'M062X'].
        basis_set : `str`
            The basis set of the calculation [default: '6-311++G(d,p)'].
        smd : `bool`
            ``True`` if SMD implicit solvent model for water required.
            Otherwise gas phase calcualtion.
        modred_input: `str`
            Modredundant input lines.
            Otherwise None type if no Modredundant input required [default: None].
        geometry: `np array`
            x, y, z coordinates for each atom in molecule.
            Otherwise None type if molecule to be read from chk [default: None].
        atom_ids: `list`
            The atom ID for ech atom in the molecule.
            Otherwise None type if molecule to be read from chk [default: None].
        charge: `int`
            The charge of the molecule.
            Otherwise None type if molecule spec to be read from chk [default: None].
        multiplicity: `int`
            The multiplicity of the molecule.
            Otherwise None type if molecule spec to be read from chk [default: None].

        """
        # Set output file, computational resources and job title
        self.output_file = self.set_output_file(output_file)
        self.nproc, self.mem = nproc, mem
        self.job_title = self.set_job_title(job_type)

        # Set molecule specification or allchk/chk flags
        if all([i is None for i in [geometry, atom_ids, charge, multiplicity]]):
            self.allchk = True
            self.chk = False
        else:
            self.molecule_spec = self.set_molecule_spec(geometry, atom_ids, charge, multiplicity)
            self.allchk = False
        
        # Set job specification and modredundant input
        self.job_spec = self.set_job_spec(method, basis_set, job_type, modred_input, smd)
        if modred_input is not None:
            self.modred_input = modred_input.split(',')
        else:
            self.modred_input = modred_input


    def set_output_file(self, output_file):
        """
        Remove '.com' extension from output file if present.

        Parameters
        ----------
        output_file : `str`
            The file name/path of the output com file.

        Returns
        -------
        output_file : `str`
            Processed file name/path of the output com file.

        """
        if '.com' in output_file:
            return output_file.split('.')[0]
        else:
            return output_file


    def _set_job_type(self, job_type):
        """
        Set keywords needed for job type in the job specification.

        Parameters
        ----------
        job_type : `str`
            The type of calculation [one of: freq, fopt, opt, ts, reopt, scan; default: fopt].
        
        Returns
        -------
        job_input: `str`
            Gaussian keywords for specified job type.
        
        """
        # Dict of predefined job types and keywords
        job_type_input = {'opt': ' Opt(Tight)' , 'reopt':' Opt(Tight,RCFC) Freq', 'fopt': ' Opt(Tight) Freq', 'freq': ' Freq', 'ts': ' Opt(Tight,TS NoEigen,RCFC) Freq', 'scan': ' Opt(MaxCycles=50)'}

        # Sets job type keywords to matching dict entry or lets user enter own input
        if job_type.lower() in job_type_input:
            return job_type_input[job_type]
        elif job_type.lower() == 'own':
            job_input = ' ' + input("Enter job inputs as they would appear in Gaussian .com file:\n")
            return job_input
        else:
            raise Exception("Job type does not match known type")


    def _set_modred_keyword(self, current_job_spec):
        """Edit class attribute job_spec to add ModRedundant keyword in Opt section."""
        try: 
            # Find position of opt in job spec and split around this point
            opt_position = current_job_spec.lower().index('opt')

            # Insert modred keyword after Opt
            start = current_job_spec[:opt_position]
            end = current_job_spec[opt_position+4:]
            job_spec = start + 'Opt(ModRedundant,' + end

            return job_spec
            
        except:
            print('Opt keyword not mentioned in job spec but modredundant input given')


    def set_job_spec(self, method, basis_set, job_type, modred_input, smd):
        """
        Set job specification from all keywords.

        Parameters        
        ----------
        method : `str`
            The method of the calculation (functional, etc.) [default: 'M062X'].
        basis_set : `str`
            The basis set of the calculation [default: '6-311++G(d,p)'].
        job_type : `str`
            The type of calculation [one of: freq, fopt, opt, ts, reopt, scan; default: fopt].
        modred_input: `str`
            Modredundant input lines.
            Otherwise None type if no Modredundant input required [default: None].
        smd : `bool`
            ``True`` if SMD implicit solvent model for water required.
            Otherwise gas phase calcualtion.

        Returns
        -------
        job_spec: `str`
            The job specification line containing all Gaussian commmands for the calculation.

        """
        # Initialise job spec
        job_spec = '#P '

        # Set method and basis set and append to job spec
        job_method = method + '/' + basis_set
        job_spec += job_method

        # Add job type keywords
        job_spec += self._set_job_type(job_type)

        # Edit Opt section of job spec if ModRedundant input present
        if modred_input != None:
           job_spec = self._set_modred_keyword(job_spec)

        # Set SMD keyword
        job_spec += ' SCRF(SMD)'*(smd == True)

        # Set level of geom/guess read options
        job_spec += ' Geom(AllCheck) Guess(Read)'*(self.allchk==True)
        job_spec += ' Geom(Check) Guess(Read)'*(self.chk==True)

        # Add convergence criteria
        job_spec += ' SCF(Conver=9) Int(Grid=UltraFine)'

        return job_spec


    def set_molecule_spec(self, geometry, atom_ids, charge, multiplicity):
        """
        Process atom ids and coordinates in to format for .com file geom input.

        Parameters
        ----------
        geometry: `np array`
            x, y, z coordinates for each atom in molecule.
            Otherwise None type if molecule to be read from chk [default: None].
        atom_ids: `list`
            The atom ID for ech atom in the molecule.
            Otherwise None type if molecule to be read from chk [default: None].
        charge: `int`
            The charge of the molecule.
            Otherwise None type if molecule spec to be read from chk [default: None].
        multiplicity: `int`
            The multiplicity of the molecule.
            Otherwise None type if molecule spec to be read from chk [default: None].

        Returns
        -------
        molecule_spec: `list of str`
            The molecule specification input lines (charge and multiplicity, x y z coordinates).

        """
        # Set charge and multiplicity as first line of molecule specification
        molecule_spec = [str(charge) + ' ' + str(multiplicity)]
        
        # If x, y, z coordinates of molecule given then append to molecule specification
        if geometry is not None:
            # Set formatted line for each atom of atom id and x, y, z coordinate
            for i in range(len(atom_ids)):
                molecule_spec.append('{0:<4} {1[0]: >10f} {1[1]: >10f} {1[2]: >10f}'.format(atom_ids[i], geometry[i,:]))
            self.chk = False
        else:
            self.chk = True

        return molecule_spec


    def set_job_title(self, job_type):
        """Set job title as combination of output file name and calculation type."""
        return self.output_file + ' ' + job_type


    def write_com_file(self):
        """Write new com file."""
        # Open destination .com file
        with open(self.output_file+'.com', 'w+') as output:

            # Write comp settings - note chk file name will be same as .com file
            print('%chk={}'.format(self.output_file), file=output)
            print('%nprocshared={:d}'.format(self.nproc), file=output)
            print('%mem={:d}MB'.format(self.mem), file=output)

            # Write job spec
            print(self.job_spec + '\n', file=output)
            
            # Write title and molecule spec
            if self.allchk == False:
                print(self.job_title + '\n', file=output)
                for line in self.molecule_spec:
                    print(line, file=output)

            # Write modredundant input
            if self.modred_input != None:
                print('', file=output)
                for line in self.modred_input:
                    print(line, file=output)

            # Necessary white space to end
            print('\n\n', file=output)


##############################################################################################################
# Outside of class - functions that process information from various sources to use for writing the .com file

def parse_com_file(initial_com_file, section=2):
    """
    Parse charge and multiplicity and geometry from initial com file.
    
    Parameters
    ----------
    initial_com_file: `str`
        The file path/name of input .com file.
    section: `list of int`
        Key of target block to pull of .com file:
                            2 - Pull charge/multiplicty and geometry section [default].
                            3 - Pull modredundant input.
                            Possilibity for other pre done input (e.g. pseudopotential) but must know which section is required.

    Returns
    -------
    section_output: `list of str`
        The lines from the targetted block of the input .com file.
    
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
    Set computational resources from corresponding line in presets file.

    Parameters
    ----------
    preset: `int`
        Index of desired line (1 index) in presets file to set resources from.

    Returns
    -------
    nproc : `int`
        Number of prcoessors required for calculation.
    mem_MB : `int`
        Amount of memory required for calculation.

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


# Function to process and set molecule geometry and atom ids from input files
def geom_from_log(input_file, geom_step):

    molecule = ml.init_mol_from_log(input_file, opt_steps=geom_step)
    return molecule.geom, molecule.atom_ids

def geom_from_xyz(geom_file, geom_step):

    molecule = ml.init_mol_from_xyz(input_file)
    return molecule.geom, molecule.atom_ids

def geom_from_com(input_file, geom_step):

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

# Function to process and set molecule charge and multiplicity from input files
def cm_from_log(input_file):

    molecule = ml.GaussianLog(input_file)
    multiplicity = molecule.pull_multiplicity()
    return molecule.charge, multiplicity
    
def cm_from_com(input_file):

    molecule_spec = parse_com_file(input_file)[0]
    charge, multiplicity = [
        int(i) 
        for i in molecule_spec.split()
    ]
    return charge, multiplicity


def set_molecule_spec(input_file, pull_geometry=True, pull_cm=False, geom_step=None):

    # Use input file type to map to function for setting molecule spec
    mol_spec = []
    geometry_functions = {'log': geom_from_log, 'com': geom_from_com, 'xyz': geom_from_xyz}
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


def push_com(output_file, job_type='fopt', preset=None, nproc=20, mem=62000, method='M062X', basis_set='6-311++G(d,p)', smd=None, modred_input=None, molecule_input=None, geom_step=None, cm_input=None):
    """Create a com file class instance and write a new com file."""
    # Set computatonal resources
    if preset != None:
        nproc, mem = parse_comp_presets(preset)

    # Process cm/molecule input options to set them to appropriate values if either are None
    molecule_input, cm_input = test_input(molecule_input, cm_input)

    # If allchk then call com file with no molecule spec
    if 'allchk' in molecule_input:
        new_com_file = GaussianCom(output_file, job_type=job_type, nproc=nproc, mem=mem, method=method, basis_set=basis_set, smd=smd, modred_input=modred_input)

    else:
        # Set charge and multilpicty from file source or directly if given
        charge, multiplicity = set_molecule_spec(cm_input, pull_geometry=False, pull_cm=True, geom_step=geom_step)    

        # If chk then call com file with no atom ids or geometry
        if 'chk' in molecule_input:
            new_com_file = GaussianCom(output_file, job_type=job_type, nproc=nproc, mem=mem, method=method, basis_set=basis_set, smd=smd, modred_input=modred_input, charge=charge, multiplicity=multiplicity)
        
        # Set geometry and atom ids from source file
        else:
            geometry, atom_ids = set_molecule_spec(molecule_input, geom_step=geom_step)
            new_com_file = GaussianCom(output_file, job_type=job_type, nproc=nproc, mem=mem, method=method, basis_set=basis_set, smd=smd, modred_input=modred_input, charge=charge, multiplicity=multiplicity, geometry=geometry, atom_ids=atom_ids)

    # Write .com file
    new_com_file.write_com_file()


if __name__ == "__main__":

    '''Argparse usage and arguments'''

    usage = "usage: %(prog)s [output_file] [args]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("output_file", type=str, help="The name of the .com file to be written [without .com]")
    parser.add_argument("-j", "--job", dest="job_type", type=str, default='fopt', 
                        choices=['freq', 'opt', 'reopt', 'fopt', 'scan', 'ts', 'own'],
                        help="Gaussian job type, currently available: opt, freq, fopt [opt+freq], reopt (opt+freq from chk), scan (opt(ModRedundant)), ts (TS Opt), own (enter own arguments)")
    parser.add_argument("-m", "--method", dest="method", type=str, default='M062X',
                        help="The method to be used, give the correct gaussian keyword")
    parser.add_argument("-b", "--basis_set", dest="basis_set", type=str, default='6-311++G(d,p)',
                        help="The basis set to be used, give the correct gaussian keyword")
    parser.add_argument("-g", "--geom", dest="geom_input", nargs='*', default=['allchk'],
                        help="Input source file for molecule geometry and atom ids (and charge/multiplicity if same source) or chk/allchk. A specific optimisation step can be provided if log file")
    parser.add_argument("-c", "--cm", dest="cm_input", nargs='*', 
                        help="Input source for charge and multiplicity of molecule, either file source or values in order of charge multiplicity (e.g. 0 1)")
    parser.add_argument("--mod", dest="modred_input", type=str,
                        help="ModRedundant input, each input line entered as a csv if multiple")
    parser.add_argument("--smd", dest="smd", action='store_true',
                        help="Flag whether to include SMD keyword for solvation or not, set to true for input")
    parser.add_argument("-p", dest="preset", nargs=1, type=int,
                        help="Preset flag to set required prcoessors and mem from presets file")
    parser.add_argument("--nproc", dest="nproc", nargs=1, type=int, default=20,
                        help="Number of processors")  
    parser.add_argument("--mem", dest="mem", nargs=1, type=int, default=62000,
                        help="Memory requirement of job (MB)")

    args = parser.parse_args()
    
    # Process geom input if optstep is given.
    if len(args.geom_input) == 2:
        geom_input, geom_step = args.geom_input
    else:
        geom_input = args.geom_input[0]
        geom_step = None

    # Process cm input for either raw input or file name.
    if args.cm_input != None and len(args.cm_input) == 1:
        cm_input = args.cm_input[0]
    else:
        cm_input = args.cm_input
    
    # Call method to create and write com file.
    push_com(args.output_file, job_type=args.job_type, preset=args.preset, nproc=args.nproc, mem=args.mem, method=args.method, basis_set=args.basis_set, smd=args.smd, modred_input=args.modred_input, molecule_input=geom_input, geom_step=geom_step, cm_input=cm_input)