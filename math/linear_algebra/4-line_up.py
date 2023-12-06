#!/usr/bin/env python3
"""
This module contains a function that perform a add operation
on two line array(vectors)

Example:
>>> arr1 = [1, 2, 3, 4]
... arr2 = [5, 6, 7, 8]
... print(add_arrays(arr1, arr2))
... print(arr1)
... print(arr2)
... print(add_arrays(arr1, [1, 2, 3]))
[6, 8, 10, 12]
[1, 2, 3, 4]
[5, 6, 7, 8]
None

Function:
   def add_arrays(arr1, arr2)
"""

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
    for line range(len(arr1)):
        new_vector.append(arr1[line] + arr2[line])

    return new_vetor
