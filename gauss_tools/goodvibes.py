"""Module to process GoodVibes results and to add quasi-harmonic values to an existing dataframe of molecules."""

import sys
import argparse
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import phdtbtk
import molLego as ml

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

def pull_goodvibes(file_name):
    """
    Parse results from GoodVibes output file.

    Parameters
    ----------
    file_name : :class: `str`
        Name/path of goodvibes data file to be processed.

    Returns
    -------
    goodvibes_df : :pandas:`DataFrame`
        Results from GoodVibes calculations.
    calculation_properties : :class:`dict`
        Key: Value pair is the parameters of the GoodVibes 
        calculation and value.
        E.g. T, C, Scale factor.
    
    """
    # Calculation output flags: Temp; Conc; Scale factor
    property_flags = {'Temperature =': 'temp', 'Concentration =': 'conc', 
                      'scale factor': 'scale_factor', 'scaling factor of': 'scale_factor'}

    # Initialise variables
    calculation_properties = {}
    raw_results = []

    # Open file and pull thermodynamic results.
    with open(file_name, 'r') as infile:
        for line in infile:

            if 'Structure' in line:
                col_headings = line.split()
                line = infile.__next__()
                line = infile.__next__()
                while line[0] == 'o':
                    entries = [line.split()[1]]
                    entries += [float(i) for i in line.split()[2:]]
                    raw_results.append(dict(zip(col_headings, entries)))
                    line = infile.__next__()
            else:
                for flag, property in property_flags.items():
                    if flag in line:
                        calculation_properties[property] = float(
                            line.split(flag)[1].split()[0])

    # Create dataframe of results with the structure as index column
    goodvibes_df = pd.DataFrame(raw_results).set_index(['Structure'])

    # Map existing goodvibescolumn headings to new ones.
    qh_new_headings = {'T.S':'TS', 'G(T)':'G', 'T.qh-S':'TqhS', 'qh-G(T)':'qhG'}
    if 'qh-H' in list(goodvibes_df.columns):
        qh_new_headings['qh-H']  = 'qhH'
    goodvibes_df = goodvibes_df.rename(columns=qh_new_headings)

    # Remove repeated rows from goodvibes dataframe.
    goodvibes_df = goodvibes_df.drop_duplicates()
    # Convert to kJ/mol.
    goodvibes_df = goodvibes_df*2625.5

    return goodvibes_df, calculation_properties

def process_file_name(file_name):
    return file_name.split('/')[-1][:-4]

def process_dataframes(goodvibes_df, molecule_df):
    """
    Combine initial molecule data with GoodVibes results.

    Parameters
    ----------
    goodvibes_df : :pandas:`DataFrame`
        Results from GoodVibes calculations.
    molecule_df : :pandas:`DataFrame`
        Exisiting molecule data.

    Returns
    -------
    molecule_df : :pandas:`DataFrame`
        Processed dataframe of molecules with GoodVibes and initial molecule results.
    
    """
    # Drop existing relative value and s/g columns from molecule dataframe.
    columns_to_remove = ['s', 'g', 'relative e', 'relative e_therm', 
                         'relative h', 'relative g']
    molecule_df = molecule_df.drop(columns_to_remove, axis=1)

    # Reset index to match goodvibes df index.
    molecule_df['index_from_file'] = molecule_df['file_name'].apply(process_file_name)
    molecule_df.set_index('index_from_file')

    # Append dataframes and calculate new realtive values.
    molecule_df = pd.concat([molecule_df, goodvibes_df], axis=1)
    relative_q_cols = [x for x in molecule_df.columns if 'qh' in x]
    molecule_df = ml.calc_relative(molecule_df, quantities=relative_q_cols)
    return molecule_df

def update_mol_goodvibes(mol, goodvibes_df):
    """
    Update thermochem values to Molecule Thermo.

    Parameters
    ----------
    mol : :Molecule:
        Molecule Thermo to add goodvibes attributes too.
    
    goodvibes_df : :pandas:`DataFrame`
        Results from GoodVibes calculations.

    Returns
    :Molecule:
        Molecule with altered thermochemistry attributes.
    
    """
    # Set file index for goodvibes results dataframe.
    mol_name = process_file_name(mol.parser.file_name)
    
    # Update G and S.
    mol.s = goodvibes_df.loc[mol_name, 'TqhS']/mol.t
    mol.g = goodvibes_df.loc[mol_name, 'qhG']

    # Update H is qh correction calculated.
    if 'qhH' in goodvibes_df.columns:
        mol.h = goodvibes_df.loc[mol_name, 'qhH']

def compare_rank(mol_data, save=None):
    """
    Plot comparison of conformer rankings across different thermodynamic values.
    
    Parameters
    ----------
    mol_data_final: :pandas:`DataFrame`
        Molecule data with GoodVibes and initial (RHHO) molecule results.
    save: `str`
        png image name.

    """
    # Set figure with subplots
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

    # Set comparison pairs and plot scatter for each one
    x_labels = ['Relative E', 'Relative E', 'Relative G']
    y_labels = ['Relative qhG', 'Relative G', 'Relative qhG']
    for i in range(3):
        fig, ax[i] = phdtbtk.plot_order(mol_data, x_labels[i], mol_data, y_labels[i], fig=fig, ax=ax[i])
        ax[i].set_ylabel('Conformer rank $\Delta$' + y_labels[i].split()[-1])
        ax[i].set_xlabel('Conformer rank $\Delta$' + x_labels[i].split()[-1])
    
    # save/show plot
    plt.tight_layout()
    if save != None:
        fig.savefig(save + '_comp_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    
    """CL arguments for running goodvibes analysis/appending dataframe."""

    usage = "usage: %(prog)s [goodvibes_file] [mol_file]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("goodvibes_file", type=str, 
                        help="Goodvibes results file.")
    parser.add_argument("molecule_input_file", type=str, 
                        help="Existing molecule datafile (.csv) or conf file of molecules.")
    parser.add_argument("-s", "--save", dest="save", type=str, 
                        help="File name to save results to (minus .csv extension)")

    args = parser.parse_args()
    
    # Process goodvibes results.
    gv_df, calc_prop = pull_goodvibes(args.goodvibes_file)

    # Workflow one - work with molecule dataframe.
    mol_df = process_mol_input(args.molecule_input_file, as_dataframe=True)
    mol_gv_df = process_dataframes(gv_df, mol_df)

    # # Workflow two - work with molecules (all with goodvibes results).
    # mol_names, mols = process_mol_input('vxoh_ax.conf', as_dataframe=False)
    # mols_gv = []
    # for i, mol in enumerate(mols):
    #     try:
    #         mols_gv.append(update_mol_goodvibes(mol, gv_df))
    #     except:
    #         pass






    

    

    
