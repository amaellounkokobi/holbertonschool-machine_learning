#!/usr/bin/env python3
"""
This module contains a function that concatenates
two matrices along a specific axis

Examples:
   >>> mat1 = np.array([[11, 22, 33], [44, 55, 66]])
   ... mat2 = np.array([[1, 2, 3], [4, 5, 6]])
   ... print(np_cat(mat1, mat2))
   ... print(np_cat(mat1, mat2, axis=1))
   [[11 22 33]
    [44 55 66]
    [ 1  2  3]
    [ 4  5  6]]
   [[11 22 33  1  2  3]
    [44 55 66  4  5  6]]

Functions:
   def np_cat(mat1, mat2, axis=0)

Libraries:
   Numpy

"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    This function def np_cat(mat1, mat2, axis=0):
    concatenates two matrices along a specific axis.

    Args:
       mat1: Numpy n dimension matrix
       mat2: Numpy n dimension matrix

    Returns
       new_matrix: the new concatenated matrix

    """
    new_matrix = np.concatenate((mat1, mat2), axis)

    return new_matrix
