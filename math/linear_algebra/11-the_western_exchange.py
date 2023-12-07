#!/usr/bin/env python3
"""
This module contains :
- A function def np_transpose(matrix): that transposes matrix:

Examples:
   mat1 = np.array([1, 2, 3, 4, 5, 6])
   mat2 = np.array([])
   mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                    [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
   print(np_transpose(mat1))
   print(mat1)
   print(np_transpose(mat2))
   print(mat2)
   print(np_transpose(mat3))
   print(mat3)0

Functions:
   def np_transpose(matrix)
"""

def np_transpose(matrix):
    """
    This function transpose a N dimensional array

    Args:
       matrix: Numpy array.

    Returns:
       A new numpy N dimension array.
    """

    return matrix.T
