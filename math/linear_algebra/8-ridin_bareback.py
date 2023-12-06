#!/usr/bin/env python3
"""
This module contains 
- A function that performs a matrix multiplication
- A function that return the shapes of a Matrix

Example:
   >>> mat1 = [[1, 2],
   ...        [3, 4],
   ...        [5, 6]]
   ... mat2 = [[1, 2, 3, 4],
   ...         [5, 6, 7, 8]]
   ... print(mat_mul(mat1, mat2))
   [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]

functions:
   def mat_mul(mat1, mat2)
"""


def matrix_shape(matrix):
    """
    This function calculates the shape of a matrix

    args:
       matrix(array):

    Returns:
      shape(list): Shape of the matrix
    """

    shape = []
    while type(matrix) is list:
        if len(matrix):
            shape.append(len(matrix))
        else:
            shape.append(0)
        if matrix:
            matrix = matrix[0]
        else:
            break

    return shape


def mat_mul(mat1, mat2):
    """
    This function perform a matrix multiplication

    Args:
       mat1: 2D matrix int/float
       mat2: 2D matrix int/float

    Returns:
       A new matrix
    """

    """Multiplication condition"""

    sp_mat1 = matrix_shape(mat1)
    sp_mat2 = matrix_shape(mat2)

    lin_mat1 = sp_mat1[0]
    lin_mat2 = sp_mat2[0]
    col_mat1 = sp_mat1[1]
    col_mat2 = sp_mat2[1]

    if col_mat1 != lin_mat2:
        return None

    """Creating the new matrix """
    new_matrix = [[0] * col_mat2 for ite in range(lin_mat1)]

    """Fill the new matrix with values"""
    result = 0

    for i_lin in range(lin_mat1):
        for i_col in range(col_mat2):
            i_lin_col = len(mat1[i_lin])
            for index in range(i_lin_col):
                calc = mat1[i_lin][index] * mat2[index][i_col]
                result = result + calc
            new_matrix[i_lin][i_col] = result
            result = 0

    return new_matrix
