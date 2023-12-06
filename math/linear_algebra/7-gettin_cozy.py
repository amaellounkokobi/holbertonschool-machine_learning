#!/usr/bin/env python3
"""
This module contains :
- A function taht concatanate two matrices.
- A function that return the shapes of a matrix.
- A function that makes a full copy of a matrix

Examples:
>>> mat1 = [[1, 2], [3, 4]]
... mat2 = [[5, 6]]
... mat3 = [[7], [8]]
... mat4 = cat_matrices2D(mat1, mat2)
... mat5 = cat_matrices2D(mat1, mat3, axis=1)
... print(mat4)
... print(mat5)
... mat1[0] = [9, 10]
... mat1[1].append(5)
... print(mat1)
... print(mat4)
... print(mat5)
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]
[[9, 10], [3, 4, 5]]
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]

Functions:
   def cat_matrices2D(mat1, mat2, axis=0):
   def matrix_shape(matrix):

"""


def matrix_shape(matrix):
    """
    That calculates the shape of a matrix.

    args:
       matrix(array):

    Returns:
      shape(list): Shape of the matrix.
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



def matrix2D_isEmpty(matrix):
    """
    A function that returns true if the matrix is empty

    Args:
       matrix/ 2D matrix of int/float

    Returns:
       True/False

    """
    result = False

    for line in matrix:
        if line:
            result = result or False
        else:
            result = result or True

    return result


def matrix2D_copy(matrix):
    """
    A function that returns a copy of a 2Dmatrix

    Args:
       matrix/ 2D matrix of int/float

    Returns:
       new matrix: 2D matrix

    """
    new_matrix = []

    for line in matrix:
        new_line = []
        for column in line:
            new_line.append(column)
        new_matrix.append(new_line)

    return new_matrix


def cat_matrices2D(mat1, mat2, axis=0):
    """
    A function def cat_matrices2D(mat1, mat2, axis=0):
    that concatenates two matrices along a specific axis.

    Args:
       mat1: 2D matrix of int/float
       mat2: 2D matrix of int/float

    Returns
       new_matrix: the new concatenated 2D matrix
       None: If the two matrices cannot be concatenated

    """
    new_matrix = matrix2D_copy(mat1)
    
    sh_mat1 = matrix_shape(mat1)
    sh_mat2 = matrix_shape(mat2)

    if axis == 0:
        if sh_mat1[1] == sh_mat2[1]:
            for index in range(len(mat2)):
                new_matrix.append(mat2[index])
        else:
            return None
        
    elif axis == 1:
        if sh_mat1[0] == sh_mat2[0]:
            for index in range(len(mat1)):
                for index2 in range(len(mat2[index])):
                    new_matrix[index].append(mat2[index][index2])
        else:
            return None
        
    else:
        pass

    return new_matrix
