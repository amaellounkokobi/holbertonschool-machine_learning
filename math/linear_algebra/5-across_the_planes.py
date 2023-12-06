#!/usr/bin/env python3
"""
This module contains a function that adds two matrices and
a function that perform a add operation on two line array(vectors)

Example:
   >>> mat1 = [[1, 2], [3, 4]]
   ... mat2 = [[5, 6], [7, 8]]
   ... print(add_matrices2D(mat1, mat2))
   ... print(mat1)
   ... print(mat2)
   ... print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
   [[6, 8], [10, 12]]
   [[1, 2], [3, 4]]
   [[5, 6], [7, 8]]
   None

function:
   def matrix_shape(matrix):
   def add_matrices2D(mat1, mat2):

"""


def matrix_shape(matrix):
    """
    That calculates the shape of a matrix

    args:
       matrix(array):

    Returns:
      shape(list): Shape of the matrix
      None: if the matrice is empty or if the shape is empty
    """

    shape = []

    if len(matrix) == 0:
        return None
    
    while 42:
        shape.append(len(matrix))
        if type(matrix[0]) is list:

            matrix = matrix[0][0:len(matrix[0])]
        else:
            break

    if len(shape) == 0:
        return None
        
    return shape


def add_matrices2D(mat1, mat2):
    """
    A function that adds two matrices element-wise

    Args:
       mat1():2Dmatrice
       mat2():2Dmatrice

    Returns:
       A new matrice
    """

    mat1_shape = matrix_shape(mat1)
    mat2_shape = matrix_shape(mat2)

    if mat1_shape != mat2_shape:
        return None

    if len(mat1) > 0:
        len_line = mat1_shape[0]
        len_column = mat1_shape[1]
    else:
        return []

    new_matrix = []

    for line in range(len_line):
        new_line = []
        for column in range(len_column):
            new_line.append(mat1[line][column] + mat2[line][column])
        new_matrix.append(new_line)

    return new_matrix
