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
    err1 = 'n must be a positive integer'
    err2 = 'x must be an integer that is greater than or equal to 0'
    err3 = 'x cannot be greater than n'
    err4 = 'P must be a 1D numpy.ndarray'
    err5 = 'All values in P must be in the range'

    if not isinstance(n, int) or n <=  0:
        raise ValueError(err1)

    if not isinstance(x, int) or x < 0:
        raise ValueError(err2)

    if x > n:
        raise ValueError(err3)

    if not isinstance(P,np.ndarray) or P.ndim != 1:
        raise TypeError(err4)

    if not (np.all(P > 0) and np.all(P < 1)):
        raise ValueError(err5)

    q = 1 - P
    n_fail = n - x
    coef = np.math.factorial(n) / (np.math.factorial(x)
                                   * np.math.factorial(n_fail))
    power_success = pow(P, x)
    power_fail = pow(q, n_fail)

    L = coef * power_success * power_fail


    return L
