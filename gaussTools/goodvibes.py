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


'''Script to process GoodVibes results and to add quasi-harmonic values to an existing dataframe of molecules'''


def parse_goodvibes(input_file):

    '''Parses results from a goodvibes output file

    Parameters:
     input_file: str - name/location of goodvibes data file to be processed

    Returns:
     goodvibes_data: pd DataFrame - dataframe containing the results from a goodvibes calculations
     calculation_properties: dict - details of goodvibes calcualtions (temperature, concentration and scale factor is used)
    '''

    # Dict for flags of calculation output wanted: Temperature; Concentration; Scale factor
    property_flags = {'Temperature =': 'temp', 'Concentration =': 'conc', 'scale factor': 'scale_factor', 'scaling factor of': 'scale_factor'}

    # Initialise variables
    calculation_properties = {}
    raw_results = []

    # Open file and pull thermodynamic results
    with open(input_file, 'r+') as input:
        for line in input:
            for flag, property in property_flags.items():
                if flag in line:
                    calculation_properties[property] = float(line.split(flag)[1].split()[0])

            if 'Structure' in line:
                col_headings = line.split()
                line = input.__next__()
                line = input.__next__()
                while line[0] == 'o':
                    entries = [line.split()[1]]
                    entries += [float(i) for i in line.split()[2:]]
                    raw_results.append(dict(zip(col_headings, entries)))
                    line = input.__next__()

    # Create dataframe of results with the structure as index column
    goodvibes_data = pd.DataFrame(raw_results).set_index(['Structure'])

    return goodvibes_data, calculation_properties


def process_file_name(file_name):
        return file_name.split('/')[-1][:-4]


def process_dataframes(goodvibes_index, mol_data):

    '''Sets a Structure column as a new index for the existing molecule data that corresponds to goodvibes dataframe and sorts order to match goodvibes dataframe

    Parameters:
     goodvibes_index: pd DataFrame index - The index ('Structure') used in the goodvibes dataframe 
     mol_data: pd DataFrame - dataframe contanining the initial molecule results

    Returns:
     mol_data: pd DataFrame - processed dataframe consisting only of entries with corresponding goodvibes data
    '''

    # Create a new column of the corresponding goodvibes structure
    mol_data['Structure'] = mol_data['File'].apply(process_file_name)

    # Subset mol_data frame to the entries present in the goodvibes dataframe
    mol_data = mol_data.loc[mol_data['Structure'].isin(goodvibes_index)]

    # Set index and order to match goodvibes dataframe
    mol_data = mol_data.set_index(['Structure'])
    
    # Need to fix here to prevent new columns being added if not included 
    mol_data = mol_data.reindex(index=goodvibes_index)

    return mol_data


def check_thermo_results(mol_data, goodvibes_data, calculation_properties):

    '''Checks that RRHO thermodynamic quantities are the same across the goodvibes and gaussian thermochemistry results. If not then prints a warning. Copies of the DataFrames are created to avoid changing the data in place.

    Only the electronic E, H, ZPE are checked as TS and G will not match if solvation calculation

    Parameters:
     mol_data: pd DataFrame - dataframe of molecule data from gaussian log file
     goodvibes_data: pd DataFrame - dataframe containing the results from a goodvibes calculations
     calculation_properties: Dict - properties of the GoodVibes calculation including concentration, temperature and scale factor

    ''' 
    # Convert all quantities to kj/mol
    goodvibes_comp_data = goodvibes_data[:]*2625.5
    mol_comp_data = mol_data
    mol_comp_data['E SCF (h)'] *= 2625.5

    mol_cols = ['ZPE', 'E SCF (h)', 'H', 'G']
    goodvibes_cols = ['ZPE', 'E', 'H', 'G(T)']

    diff = pd.DataFrame()
    
    # Try comparison with unscaled gaussian data
    comp = phdtbtk.data_comp(mol_comp_data, goodvibes_comp_data, df_cols_1=mol_cols, df_cols_2=goodvibes_cols, tol=1e-2)
    if all(comp):
        return
    
    # Try comparison with scaled gaussian data
    elif ('scale_factor' in calculation_properties):
        scale_factor = calculation_properties['scale_factor']
        mol_comp_data['ZPE_scaled'] = scale_factor*mol_comp_data['ZPE']
        mol_comp_data['H_scaled'] = mol_comp_data['H'] - mol_comp_data['ZPE'] + mol_comp_data['ZPE_scaled']
        mol_cols = ['ZPE_scaled', 'E SCF (h)', 'H_scaled']
        comp = phdtbtk.data_comp(mol_comp_data, goodvibes_comp_data, df_cols_1=mol_cols, df_cols_2=goodvibes_cols[:-1], tol=1e-2)

    if not all(comp):
        print('Warning: goodvibes thermo results do not match original gaussian results (tolerence: ', 1e-2, ')')
        print(comp)

    return
        

def append_goodvibes(mol_data, goodvibes_data):

    '''Appends the quasi-RRHO entropy and free energy values to the original data frame.
    
    Parameters:
     mol_data: pd DataFrame - dataframe of molecule data from gaussian log file
     goodvibes_data: pd DataFrame - dataframe containing the results from a goodvibes calculations
    
    Returns:
     mol_data: pd DataFrame - updated dataframe with columns with quasi-RRHO entropy and free energy values and relative values
    '''

    # Set headings
    qh_new_headings = ['TqhS', 'qhG']
    qh_goodvibe_headings = ['T.qh-S', 'qh-G(T)']

    # Set RHHO columns in original dataframe (and change from h to kJ/mol)
    mol_data[qh_new_headings] = goodvibes_data[qh_goodvibe_headings]*2625.5

    # Calculate relative values for them
    mol_data = ml.calc_relative(mol_data, quantities=qh_new_headings)

    return mol_data


def process_goodvibes(molecule_file, goodvibes_file, new_file=False, save=None):

    '''
    Combined workflow for processing gaussian molecule data, goodvibes data, and appending the two
    
    Parameters:
     molecule_data: str - name/location of file for molecule data [csv or conf]
     goodvibes_data: str - name/location of goodvibes data file
     new_file: bool - Flag of whether results should be appended to the existing molecule data file or create a new file
     save: str - new file name to write appended data too [default: None]

    '''
    # Create initial dataframes for existing molecule results and goodvibes results
    mol_data_initial, mols = phdtbtk.process_input_file(molecule_file)
    gv_data, gv_calc_properties = parse_goodvibes(goodvibes_file)
    
    # Process mol dataframe to have shared index and order with goodvibes df
    mol_data = process_dataframes(gv_data.index, mol_data_initial)
    
    # Check if thermodynamic quantities match across the gv and gaussian results
    check_thermo_results(mol_data, gv_data, gv_calc_properties)

    # Add columns to dataframe
    mol_data_full = append_goodvibes(mol_data, gv_data)
    
    # Save as new file or rewrite old one
    if new_file == False:
        if save == None:
            save = molecule_file.split('.')[0] + '_goodvibes'
        mol_data_full.to_csv(save + '.csv')    
    else:
        mol_data_full.to_csv(molecule_file)


if __name__ == "__main__":
    
    '''
    CL arguments for running goodvibes analysis/appending dataframe.
    '''

    usage = "usage: %(prog)s [goodvibes_data] [mol_data]"
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("goodvibes_file", type=str, 
                        help="Goodvibes results file")
    parser.add_argument("molecule_data", type=str, 
                        help="Existing datafile (.csv) of molecule data or conf file ofmolecules to append goodvibes results to be appended too")
    parser.add_argument("-n", "--new", dest="new_file", action='store_false', 
                        help="Flag of whether a new file should be created with the results [default] or append to the existing molecule_data file")
    parser.add_argument("-s", "--save", dest="save", type=str, 
                        help="Name of file to save new results too (minus .csv extension)")
    args = parser.parse_args()
    
    # Call process workflow to add goodvibes data to existing molecules/molecule dataframe
    process_goodvibes(args.molecule_file, args.goodvibes_file, new_file=args.new_file, save=args.save)





    

    

    
