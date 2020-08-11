import pandas as pd
import numpy as np

'''Module of methods for calling on dataframes'''


def data_comp(dataframe_1, dataframe_2, df_cols_1=None, df_cols_2=None, tol=1e-4):

    '''Function to compare two dataframes
    
    Parameters:
     dataframe_1: pd DataFrame - first dataframe to compare
     dataframe_2: pd DataFrame - second dataframe to compare
     df_cols_1: List of str - column headers of the first DataFrame to be compared [default:None - will set to all non-index columns in DataFrame]
     df_cols_2: List of str - column headers of the second DataFrame to be compared [default:None - will set to all non-index columns in DataFrame]
     tol: float - value of difference tolerence to be evaluated between the datasets

    '''

    # Set columns to compare as all dataframe columns if no subset given
    if df_cols_1 == None:
        df_cols_1 = list(dataframe_1.columns)[1:]
    if df_cols_2 == None:
        df_cols_2 = list(dataframe_2.columns)[1:]

    # Check that the same number of columns are being compared between the two dataframes
    if len(df_cols_1) != len(df_cols_2):
        raise Exception('Comparing unequal number of columns between dataframes')

    # Initalise variables
    diff = pd.DataFrame()

    # Calculate difference between dataframes for column subset
    diff = pd.DataFrame()
    for column in zip(df_cols_1, df_cols_2):
        diff[column[1]] = np.abs(dataframe_1[column[0]] - dataframe_2[column[1]])
    
    # retrun list of bools testing if column difference is below tolerence
    return list((diff.dropna() < tol).any())