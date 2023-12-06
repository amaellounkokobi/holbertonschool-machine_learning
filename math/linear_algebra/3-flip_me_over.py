#!/usr/bin/env python3
"""
This module contains a fuctnion that transpose a matrix

Examples:
   >>> mat1 = [[1, 2], [3, 4]]
   ... print(mat1)
   ... print(matrix_transpose(mat1))
   [[1, 2], [3, 4]]
   [[1, 3], [2, 4]]

Functions:
   matrix_transpose(matrix)

"""


def matrix_transpose(matrix):
    """
    This function flips over a matrix by his diagonal

    args:
       matrix(list of list):
    """
    matrix_T = []

    while len(matrix[0]) > 0:
        sliced_matrix = []
        matrix_row = []
        while len(matrix) > 0:
            matrix_row.append(matrix[0][0])
            matrix[0] = matrix[0][1:]
            sliced_matrix.append(matrix[0])
            matrix = matrix[1:]
        matrix_T.append(matrix_row)
        matrix = sliced_matrix

    return matrix_T
