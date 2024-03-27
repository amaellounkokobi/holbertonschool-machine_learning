#!/usr/bin/env python3
"""
This module contains :
Exercices on numpy
"""
import numpy as np

matrix = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1,5,3],[4,5,6],[7,8,9]]])

matrix2 =  np.array([[1,1],[1,1]])
print(matrix.shape)
print(matrix)
print('-----------------------------')
print(np.sum(matrix))
print('-----------------------------')
print(matrix[:,0:2,0:2])
print('-----------------------------')
print(matrix[:,0:2,0:2] * matrix2)
print('-----------------------------')
#creation de la matrice shaped
print('-----------------------------')
dest_mat = np.zeros(shape=(2,2,2))
print(dest_mat)

# convulution operation
convoluted = np.sum(np.sum(matrix[:,0:2,0:2] * matrix2, axis=1), axis=1, keepdims=True)
print('-----------------------------')
# add operations in all layer images
dest_mat[:, 0:1, 0] = convoluted

print('-----------------------------')
print(dest_mat)
