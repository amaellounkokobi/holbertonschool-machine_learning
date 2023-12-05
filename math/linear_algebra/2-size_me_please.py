#!/usr/bin/env python3
"""
this module contains a function that return the shapes of a Matrix


Example:
  >>> mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
  ... [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
  ... print(matrix_shape(mat2))

function:
  matrix_shape(matrix)

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
