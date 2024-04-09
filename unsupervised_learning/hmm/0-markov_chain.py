#!/usr/bin/env python3
"""
This module contains :
A function that determines the probability of a markov chain being
in a particular state after a specified number of iterations:

Function:
   def markov_chain(P, s, t=1):
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    A function that determines the probability of a markov chain
    being in a particular state after a specified number of iterations:

    Args:
       P: is a square 2D numpy.ndarray of shape (n, n)
       representing the transition matrix

       P[i, j]: is the probability of transitioning
       from state i to state j

       n: is the number of states in the markov chain

       s: is a numpy.ndarray of shape (1, n) representing
       the probability of starting in each state

       t: is the number of iterations that the markov
       chain has been through is a square 2D numpy.ndarray
       of shape (n, n) representing the transition matrix

    Returns:
       A numpy.ndarray of shape (1, n) representing the
       probability of being in a specific state after t
       iterations, or None on failure
    """
    Pin = s

    for _ in range(t):
        Pin = np.matmul(Pin, P)

    return Pin
