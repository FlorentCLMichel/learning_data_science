'''
Python wrapper for the library correlation.so
'''

import ctypes as ct
import numpy as np

# C library with the bootstrap method
libc = ct.CDLL('../C/correlation.so')

# types of the arguments and return value for the function to calculate the 
# correlation coefficient
libc.corr.argtypes = [np.ctypeslib.ndpointer(dtype='double'), np.ctypeslib.ndpointer(dtype='double'), ct.c_int]
libc.corr.restype = ct.c_double

# types of the arguments and return value for the function to calculate the 
# standard deviation of the correlation coefficient
libc.std_corr.argtypes = [np.ctypeslib.ndpointer(dtype='double'), np.ctypeslib.ndpointer(dtype='double'), ct.c_int, ct.c_int]
libc.std_corr.restype = ct.c_double

# Python wrapper: correlation coefficient
def corr_coeff(x,y):
    '''
    computes the correlation coefficient between x and y
    
    return type: float
    
    x: numpy array of float64
    y: numpy array of float64
    '''
    length = min([len(x), len(y)])
    return libc.corr(x, y, length)

# Python wrapper: standard deviation of the correlation coefficient
def std_corr_coeff(x,y):
    '''
    computes the standard deviation for the correlation coefficient between 
    x and y
    
    return type: float
    
    x: numpy array of float64
    y: numpy array of float64
    '''
    length = min([len(x), len(y)])
    return libc.std_corr(x, y, length, length)

def corr_and_error(dframe, column1, column2):
    '''
    computes the correlation coefficient between the values in columns column1 
    and column2 of the dataframe dframe and its standard deviation
    The two columns are assumed to contain numeric values only.
    
    dframe: pandas dataframe
    column1: string
    column2: string
    '''
	# remove nan values and ensure the type is float64
    dframe = dframe.loc[-dframe[column1].isnull() & -dframe[column2].isnull(), [column1,column2]].astype('float64')
    x = np.array(dframe[column1], dtype='float64')
    y = np.array(dframe[column2], dtype='float64')
    return corr_coeff(x,y), std_corr_coeff(x,y)

# function to print the result in a self-explanatory way
def print_corr_and_error(dframe, column1, column2):
    phrase = 'The correlation coefficient between columns {} and {} is {:.2g} ± {:.2g}.'
    corr, std_corr = corr_and_error(dframe, column1, column2)
    print(phrase.format(column1, column2, corr, std_corr))
