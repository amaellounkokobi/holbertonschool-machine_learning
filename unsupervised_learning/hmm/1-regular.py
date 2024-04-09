#!/usr/bin/env python3
"""
This module contains :
A function that determines the steady state
probabilities of a regular markov chain

Function:

"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a
    regular markov chain

    Args:
       P is a is a square 2D numpy.ndarray of shape (n, n)
       representing the transition matrix
       P[i, j] is the probability of transitioning
       n is the number of states in the markov chain

    Returns:
      a numpy.ndarray of shape (1, n) containing
      the steady state probabilities, or None on failure
    """

    # Transition matrix as zeros
    if np.any(P == 0):
        return None
    else:
        # Calculate steady state
        PIn = np.zeros(P.shape[1])
        PIn[0] = 1
        for _ in range(100):
            PIn = np.matmul(PIn, P)
        return np.array([PIn])
