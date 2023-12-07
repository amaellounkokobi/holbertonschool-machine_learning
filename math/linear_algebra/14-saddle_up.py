#!/usr/bin/env python3
"""
This module contains:
- A function that performs matrix multiplication

Example:
   >>> mat1 = np.array([[11, 22, 33], [44, 55, 66]])
   ... mat2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   ... print(np_matmul(mat1, mat2))
   [[ 330  396  462]
    [ 726  891 1056]]

Functions:
   def np_matmul(mat1, mat2)

Libraries:
   Numpy
"""


def np_matmul(mat1, mat2):
    """
    This function performs a Matrix multiplication
    of to n dimensional matrices.

    Args:
       mat1:Numpy n dimension matrix
       mat2:Numpy n dimension matrix

    Returns
       A new matrix mat1 * mat2

    """
    result = np.matmul(mat1, mat2)

    return result
