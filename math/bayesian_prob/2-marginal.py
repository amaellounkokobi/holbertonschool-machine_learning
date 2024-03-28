#!/usr/bin/env python3
"""
This module contains :
- A function that calculates the likelihood
of obtaining this data given various
hypothetical probabilities of developing severe side effects

- A function that calculates the intersection of obtaining this
data with the various hypothetical probabilities

- A that calculates the marginal probability of obtaining the data
Function:

def likelihood(x, n, P):
def intersection(x, n, P, Pr):
def marginal(x, n, P, Pr)

"""
import numpy as np


def likelihood(x, n, P):
    """
    function def likelihood(x, n, P): that calculates the likelihood
    of obtaining this data given various hypothetical probabilities
    of developing severe side effects

    Args:

    Returns:

    """
    err1 = 'n must be a positive integer'
    err2 = 'x must be an integer that is greater than or equal to 0'
    err3 = 'x cannot be greater than n'
    err4 = 'P must be a 1D numpy.ndarray'
    err5 = 'All values in P must be in the range [0, 1]'

    if not isinstance(n, int) or n <= 0:
        raise ValueError(err1)

    if not isinstance(x, int) or x < 0:
        raise ValueError(err2)

    if x > n:
        raise ValueError(err3)

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError(err4)

    if np.any((P < 0) | (P > 1)):
        raise ValueError(err5)

    q = 1 - P
    n_fail = n - x
    coef = np.math.factorial(n) / (np.math.factorial(x)
                                   * np.math.factorial(n_fail))
    power_success = pow(P, x)
    power_fail = pow(q, n_fail)

    L = coef * power_success * power_fail

    return L


def intersection(x, n, P, Pr):
    """
    A function that calculates the intersection of
    obtaining this data with the various hypothetical
    probabilities

    Args:
       x is the number of patients that develop severe side effects
       n is the total number of patients observed
       P is a 1D numpy.ndarray containing the various
       hypothetical probabilities of developing severe side effects
       Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns:
       A 1D numpy.ndarray containing the intersection
       of obtaining x and n with each probability in P, respectively
    """
    errs = ['n must be a positive integer',
            'x must be an integer that is greater than or equal to 0',
            'x cannot be greater than n',
            'P must be a 1D numpy.ndarray',
            'All values in P must be in the range [0, 1]',
            'All values in Pr must be in the range [0, 1]',
            'Pr must be a numpy.ndarray with the same shape as P',
            'Pr must sum to 1']

    Pr_s = len(Pr)
    P_s = len(P)

    if not isinstance(n, int) or n <= 0:
        raise ValueError(errs[0])

    if not isinstance(x, int) or x < 0:
        raise ValueError(errs[1])

    if x > n:
        raise ValueError(errs[2])

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError(errs[3])

    if not isinstance(Pr, np.ndarray) or Pr_s != P_s:
        raise TypeError(errs[6])

    if np.any((P < 0) | (P > 1)):
        raise ValueError(errs[4])

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError(errs[5])

    if not np.round(sum(Pr)) == 1:
        raise ValueError(errs[7])

    # Likelihood
    L = likelihood(x, n, P)

    # Intersection
    inter = L * Pr

    return inter


def marginal(x, n, P, Pr):
    """
    A function that calculates the marginal probability of
    obtaining the data

    Args:
       x is the number of patients that develop severe side effects
       n is the total number of patients observed
       P is a 1D numpy.ndarray containing the various
       hypothetical probabilities of developing severe side effects
       Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns:
       The marginal probability of obtaining x and n
    """
    errs = ['n must be a positive integer',
            'x must be an integer that is greater than or equal to 0',
            'x cannot be greater than n',
            'P must be a 1D numpy.ndarray',
            'All values in P must be in the range [0, 1]',
            'All values in Pr must be in the range [0, 1]',
            'Pr must be a numpy.ndarray with the same shape as P',
            'Pr must sum to 1']

    Pr_s = len(Pr)
    P_s = len(P)

    if not isinstance(n, int) or n <= 0:
        raise ValueError(errs[0])

    if not isinstance(x, int) or x < 0:
        raise ValueError(errs[1])

    if x > n:
        raise ValueError(errs[2])

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError(errs[3])

    if not isinstance(Pr, np.ndarray) or Pr_s != P_s:
        raise TypeError(errs[6])

    if np.any((P < 0) | (P > 1)):
        raise ValueError(errs[4])

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError(errs[5])

    if not np.round(sum(Pr)) == 1:
        raise ValueError(errs[7])

    marg = np.sum(intersection(x, n, P, Pr))

    return marg
