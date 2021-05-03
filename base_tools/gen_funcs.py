"""Module containing general functions for phd processing and tools."""

import sys
import numpy as np
import pandas as pd

import molLego as ml

def readlines_reverse(input_file):
    """
    Read lines of a file in reverse order [edited from Filip Szczypiński].

    Parameters
    ----------
    input_file: `str`
        The file to be read

    """ 
    with open(input_file) as in_file:
        
        # Move to end of file
        in_file.seek(0, 2)
        position = in_file.tell()
        line = ''

        # Iterate over lines until top is reached
        while position >= 0:
            in_file.seek(position)
            next_char = in_file.read(1)
            if next_char == "\n":
                yield line[::-1]
                line = ''
            else:
                line += next_char
            position -= 1
        yield line[::-1]

def clean_string(dirty_string):
    """
    Remove symbols from string leaving only alphanumeric characters.

    Parameters
    ----------
    dirty_string: `str`
        String to have symbols removed from.

    Returns
    -------
    clean_string: `str`
        String with only alpha/numeric characters.

    """
    clean_string = ''
    for char in dirty_string:
        if char.isalnum():
            clean_string += char
    
    return clean_string


def remove_num(number_string):
    return ''.join([x for x in number_string if not x.isdigit()])
    