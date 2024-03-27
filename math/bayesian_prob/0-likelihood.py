#!/usr/bin/env python3
"""
This module contains :
A function that calculates the likelihood 
of obtaining this data given various 
hypothetical probabilities of developing severe side effects

Function:
def likelihood(x, n, P):
"""
import numpy as np

def likelihood(x, n, P):
    """
    function def likelihood(x, n, P): that calculates the likelihood 
    of obtaining this data given various hypothetical probabilities 
    of developing severe side effects

    """
    if n < 0:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) and x >= 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P,np.ndarray):
        raise TypeError('P must be a 1D numpy.ndarray')

    if not np.any(P > 0)  and np.any(P < 1):
        raise ValueError('All values in P must be in the range')

    L = np.math.comb(n, x) * (P ** x) * ((1 - P) ** (n - x))

    return L
