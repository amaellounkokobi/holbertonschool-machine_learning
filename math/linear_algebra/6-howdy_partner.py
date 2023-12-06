#!/usr/bin/env python3
"""
This module contains :
- a function that concatanate two lits

Examples:
>>> arr1 = [1, 2, 3, 4, 5]
... arr2 = [6, 7, 8]
... print(cat_arrays(arr1, arr2))
... print(arr1)
... print(arr2)
[1, 2, 3, 4, 5, 6, 7, 8]
[1, 2, 3, 4, 5]
[6, 7, 8]

functions:
   def cat_arrays(arr1, arr2)

"""


def cat_arrays(arr1, arr2):
    """
    A function def cat_arrays(arr1, arr2): that concatenates two arrays.

    Args:
       arr1: (list(int/float))
       arr2: (list(int/float))
    Returns:
       new_array: a concatanated list.

    """
    new_array = arr1.copy()

    for element in range(len(arr2)):
        new_array.append(arr2[element])

    return new_array
