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


def pull_goodvibes(file_name):
    """
    Parse results from GoodVibes output file.

    Parameters
    ----------
    file_name: `str`
        Name/location of goodvibes data file to be processed.

    Returns
    -------
    goodvibes_data: :pandas:`DataFrame`
        Results from GoodVibes calculations.
    calculation_properties: `dict`
        Key: Value pair is the parameters of the GoodVibes calculation and value.
        E.g. T, C, Scale factor.
    
    """
    # Dict for flags of calculation output wanted: Temperature; Concentration; Scale factor
    property_flags = {'Temperature =': 'temp', 'Concentration =': 'conc', 'scale factor': 'scale_factor', 'scaling factor of': 'scale_factor'}

    # Initialise variables
    calculation_properties = {}
    raw_results = []

    # Open file and pull thermodynamic results
    with open(file_name, 'r') as infile:
        for line in infile:
            for flag, property in property_flags.items():
                if flag in line:
                    calculation_properties[property] = float(line.split(flag)[1].split()[0])

                if 'Structure' in line:
                    col_headings = line.split()
                    line = infile.__next__()
                    line = infile.__next__()
                    while line[0] == 'o':
                        entries = [line.split()[1]]
                        entries += [float(i) for i in line.split()[2:]]
                        raw_results.append(dict(zip(col_headings, entries)))
                        line = infile.__next__()

    # Create dataframe of results with the structure as index column
    goodvibes_data = pd.DataFrame(raw_results).set_index(['Structure'])

    return goodvibes_data, calculation_properties

def process_file_name(file_name):
    return file_name.split('/')[-1][:-4]

def process_dataframes(goodvibes_data, mol_data_initial):
    """
    Combine initial molecule data with GoodVibes results.

    Parameters
    ----------
    goodvibes_data : :pandas:`DataFrame`
        Results from GoodVibes calculations.
    mol_data_initial : :pandas:`DataFrame`
        Exisiting molecule data.

    Returns
    -------
    mol_data_final : :pandas:`DataFrame`
        Processed dataframe of molecules with GoodVibes and initial molecule results.
    
    """
    # Set new column headings and existing goodvibes headings for TS, G and qh quantities.
    qh_new_headings = ['TS', 'G', 'TqhS', 'qhG']
    qh_goodvibes_headings = ['T.S', 'G(T)', 'T.qh-S', 'qh-G(T)']
    if 'qh-H' in list(goodvibes_data.columns):
        qh_new_headings.append('qhH')
        qh_goodvibes_headings.append('qh-H')
    
    # Set column headings to be used from initial molecule dataframe.
    columns_to_remove = ['G', 'S', 'Relative E', 'Relative H', 'Relative G']
    mol_data_columns = list(mol_data_initial.columns[~mol_data_initial.columns.isin(columns_to_remove)])

    # Initialise a new dataframe with the new columns.
    mol_data_final = pd.DataFrame(columns=(mol_data_columns + qh_new_headings))

    # Remove repeated rows from goodvibes dataframe.
    goodvibes_data = goodvibes_data.drop_duplicates()

    # Set new rows for each molecule with initial mol and goodVibes data.
    for i in mol_data_initial.index:    
        # Create starting dict for entry
        new_mol = (mol_data_initial.loc[i, mol_data_columns]).to_dict()
        new_mol.update({i:0.0 for i in qh_new_headings})

        # Extract each file name and format as GoodVibes index.
        for file_name in mol_data_initial.loc[i, 'File'].split(','):
            file_name = process_file_name(file_name)
            
            # Locate molecule in GoodVibes data and set qh results.
            try:
                for quantity in zip(qh_new_headings, qh_goodvibes_headings):
                    new_mol[quantity[0]] += (goodvibes_data.loc[file_name, quantity[1]]*2625.5)
            except:
                print(file_name, ' not found in GoodVibes results.')
        
        # Set row for molecule/s in new data frame.
        mol_data_final.loc[i] = new_mol

    # Remove molecules without GoodVibes results.
    mol_data_final = mol_data_final[np.abs(mol_data_final['qhG']).gt(0)]
    
    # Calculate relative values for E, qhG (and qhH)
    mol_data_final = ml.calc_relative(mol_data_final, quantities=(['E'] + qh_new_headings[1:]))

    return mol_data_final


def update_reaction_profile(reaction_file, mol_data, save=None):
    """
    Create new reaction profile using GoodVibes data.

    Parameters
    ----------
    reaction_file: `str`
        Name/location of file for reaction .conf file with reaction molecules in.
    mol_data: :pandas:`DataFrame`
        Processed dataframe of molecules with GoodVibes and original molecule results.
    save: `str` 
        Output file name to write final data to. [default: None]

    Returns
    -------
    reaction_profile_data: :pandas:`DataFrame`
        Results for all reaction paths in reaction profile.

    """
    # Construct reaction paths from .conf file.
    reaction_paths = ml.construct_reaction_path(reaction_file)
    
    # Construct reaction profile.
    reaction_profile_data = ml.construct_reaction_profile(mol_data, reaction_paths, save=save, quantity=['qhH', 'qhG'])
    
    # Save new dataframe.
    if save == None:
        save = reaction_file.split('.')[0]
    reaction_profile_data.to_csv(save + '_gv_rprofile.csv')    

    print(reaction_profile_data[['Relative qhH', 'Relative qhG']])
    return reaction_profile_data


# def check_thermo_results(mol_data, goodvibes_data, calculation_properties):
#     """
#     Check consistency between GoodVibes and Guassian RHHO thermodynamic values.

#     Only the electronic E, H, ZPE are checked as TS and G will not match if solvation calculation. 
#     A warning is printed if the values are not consistant.
    
#     Parameters
#     ---------
    # mol_data: :pandas:`DataFrame`
    #     The exisiting molecule data.
    # goodvibes_data: :pandas:`DataFrame`
        # Results from GoodVibes calculations.
#     calculation_properties: `dict`
#         Key: Value pair is the parameters of the GoodVibes calculation and value.
#         E.g. T, C, Scale factor.

#     """
#     # Convert all quantities to kj/mol
#     goodvibes_comp_data = goodvibes_data[:]*2625.5
#     mol_comp_data = mol_data
#     mol_comp_data['E SCF (h)'] *= 2625.5

#     mol_cols = ['ZPE', 'E SCF (h)', 'H', 'G']
#     goodvibes_cols = ['ZPE', 'E', 'H', 'G(T)']

#     diff = pd.DataFrame()
    
#     # Try comparison with unscaled gaussian data
#     comp = phdtbtk.data_comp(mol_comp_data, goodvibes_comp_data, df_cols_1=mol_cols, df_cols_2=goodvibes_cols, tol=1e-2)
#     if all(comp):
#         return
    
#     # Try comparison with scaled gaussian data
#     elif ('scale_factor' in calculation_properties):
#         scale_factor = calculation_properties['scale_factor']
#         mol_comp_data['ZPE_scaled'] = scale_factor*mol_comp_data['ZPE']
#         mol_comp_data['H_scaled'] = mol_comp_data['H'] - mol_comp_data['ZPE'] + mol_comp_data['ZPE_scaled']
#         mol_cols = ['ZPE_scaled', 'E SCF (h)', 'H_scaled']
#         comp = phdtbtk.data_comp(mol_comp_data, goodvibes_comp_data, df_cols_1=mol_cols, df_cols_2=goodvibes_cols[:-1], tol=1e-2)

#     if not all(comp):
#         print('Warning: goodvibes thermo results do not match original gaussian results (tolerence: ', 1e-2, ')')
#         print(comp)

#     return


def process_goodvibes(molecule_file, goodvibes_file, save=None):
    """
    Workflow for processing GoodVibes data and appending to Gaussian Molecule data.
    
    Parameters
    ----------
    molecule_file: `str`
        Name/location of file for molecule data [csv or conf] of dataframe of molecules.
    goodvibes_fle: `str`
        Name/location of goodvibes data file.
    save: `str` 
        Output file name to write final data to. [default: None]

    Returns:
    mol_data_final: :pandas:`DataFrame`
        Processed dataframe of molecules with GoodVibes and initial molecule results.

    """
    # Set initial Molecule and GoodVibes dataframes.
    mol_data_initial, mols = phdtbtk.process_input_file(molecule_file)
    gv_data, gv_calc_properties = pull_goodvibes(goodvibes_file)

    # Process mol dataframe to have shared index and order with goodvibes df
    mol_data_final = process_dataframes(gv_data, mol_data_initial)
    
    # Save new dataframe.
    if save == None:
        save = goodvibes_file.split('.')[0]
    mol_data_final.to_csv(save + '.csv')    

    return mol_data_final, mols


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
    parser.add_argument("molecule_file", type=str, 
                        help="Existing molecule datafile (.csv) or conf file of molecules.")
    parser.add_argument("-s", "--save", dest="save", type=str, 
                        help="File name to save results to (minus .csv extension)")
    parser.add_argument("-r", "--reaction", dest="reaction_file", type=str,
                        help="Original molecule .conf file containing reaction path information.")
    args = parser.parse_args()
    
    # Call process workflow to add goodvibes data to existing molecules/molecule dataframe.
    mol_data = process_goodvibes(args.molecule_file, args.goodvibes_file, save=args.save)

    if args.reaction_file != None:
        reaction_profile_data = update_reaction_profile(args.reaction_file, mol_data, save=args.save)







    

    

    
