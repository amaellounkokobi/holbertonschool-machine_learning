#!/usr/bin/env python3
"""
This module contains a function that calculates the normalization
 (standardization) constants of a matrix:

Function:
   def normalization_constants(X):
"""


def normalization_constants(X):
    """
    function def normalization_constants(X): that calculates
    the normalization (standardization) constants of a matrix

    Args:
       X is the numpy.ndarray of shape (m, nx) to normalize
       m is the number of data points
       nx is the number of features
       Returns: the mean and standard deviation of each feature, respectively
    
    """
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)

    return mean, std
