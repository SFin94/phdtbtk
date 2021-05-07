"""Module with LAMMPS parser."""
import sys
import pandas as pd

class LAMMPSLog():

    def __init__(self, file_name):

        self.file_name = file_name
        thermo_flag = 'Step'

        with open(self.file_name) as in_file:
            for el in in_file:
                if thermo_flag in el:
                    # Set property list and initialise dataframe.
                    self.properties = el.split()
                    self.therm_df = pd.DataFrame(columns=self.properties[1:])
                    el = next(in_file)
                    self.therm_df = pd.concat([self.therm_df, 
                        self.pull_therm(in_file, el)])

    def pull_therm(self, in_file, current_line):
        
        therm_steps = []
        while 'Loop time' not in current_line:
            therm_steps.append(dict(zip(self.properties, 
                                        current_line.split())))
            current_line = next(in_file)

        # Set and proces dataframe. 
        therm_df = pd.DataFrame(therm_steps, columns=self.properties)
        therm_df = therm_df.set_index('Step')
        therm_df = therm_df.astype(float)
        return therm_df

