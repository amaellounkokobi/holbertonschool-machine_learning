#!/usr/bin/env python3
"""
This module contains:
- a function that adds two matrices and
- a function that perform a add operation on two line array(vectors)
- a function that return the shapes of a Matrix
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
   def matrix_shape(matrix):

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
        if matrix[0]:
            if type(matrix[0]) is list:
                matrix = matrix[0][0:len(matrix[0])]
            else:
                break
        else:
            return []

    return shape


def add_arrays(arr1, arr2):
    """
    A function def add_arrays(arr1, arr2): that adds
    two arrays element-wise:

    args:
       arr1(list(int/float):list of in float

    Returns:
       new_list: arr1 + arr2
       None: if the arr1 and arr2 are of different shapes
    """
    new_vector = []

    if len(arr1) != len(arr2):
        return None

    for line in range(len(arr1)):
        new_vector.append(arr1[line] + arr2[line])

    return new_vector


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

    new_matrix = []

    for line in range(len(mat1)):
        new_line = add_arrays(mat1[line], mat2[line])
        if new_line is not None:
            new_matrix.append(new_line)

    return new_matrix
