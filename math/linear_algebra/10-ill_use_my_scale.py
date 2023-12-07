#!/usr/bin/env python3
"""
This module contains a function that calculate the shape of a numpy.array

Example:
   >>> mat1 = np.array([1, 2, 3, 4, 5, 6])
   ... mat2 = np.array([])
   ... mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
   ...               [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
   ... print(np_shape(mat1))
   ... print(np_shape(mat2))
   ... print(np_shape(mat3))
   (6,)
   (0,)
   (2, 2, 5)
"""


def np_shape(matrix):
    """ This function calculates the shape
    of a Numpy array

    Args:
       matrix:Any numpy.array

    Returns:
       shape: A tuple of the matrix shape

    """
    return matrix.shape
