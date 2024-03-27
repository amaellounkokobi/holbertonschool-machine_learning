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



def fact(value):
    """                                                                                                                                                                                                
    This method calculate the factoriel                                                                                                                                                                
    of a number                                                                                                                                                                                                                                                                                                                                                                                                   
    Args:                                                                                                                                                                                              
        value(int):Value of factoriel                                                                                                                                                                   
    """
    result = 1

    while value >= 1:
        result *= value
        value -= 1

    return result


def likelihood(x, n, P):
    """
    function def likelihood(x, n, P): that calculates the likelihood
    of obtaining this data given various hypothetical probabilities
    of developing severe side effects

    """
    if not isinstance(n, int) or n <=  0:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or x < 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P,np.ndarray):
        raise TypeError('P must be a 1D numpy.ndarray')

    if not (np.any(P > 0) and np.any(P < 1)):
        raise ValueError('All values in P must be in the range')

    q = 1 - P
    n_fail = n - x
    coef = np.math.factorial(n) / (np.math.factorial(x)
                                   * np.math.factorial(n_fail))
    power_success = pow(P, x)
    power_fail = pow(q, n_fail)
        
    L = coef * power_success * power_fail


    return L
