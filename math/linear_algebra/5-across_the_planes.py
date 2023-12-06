#!/usr/bin/env python3
"""
This module contains a function that adds two matrices

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
    """

    shape = []

    while 42:
        shape.append(len(matrix))
        if type(matrix[0]) is list:

            matrix = matrix[0][0:len(matrix[0])]
        else:
            break

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

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    for line in mat1:
        new_line = []
        for column in line:
            new_line.append(mat1[line][column] + mat2[line][column])
        new_matrix.append(new_line)

    return new_matrix
