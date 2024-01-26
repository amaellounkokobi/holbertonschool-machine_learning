#!/usr/bin/env python3
"""

This module contains :

A function that determines if you should stop gradient descent early
Function:

   def early_stopping(cost, opt_cost, threshold, patience, count):
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Function that determines if you should stop gradient descent early

    Args:
    cost: is the current validation cost of the neural network
       opt_cost: is the lowest recorded validation cost of the neuron
       threshold: is the threshold used for early stopping
       patience: is the patience count used for early stopping
       count: is the count of how long the threshold has not been met
    Returns:
       a boolean of whether the network should be stopped
       early, followed by the updated count
    """
    if count + 1 >= patience:
        return True, count + 1
    else:
        return False, count + 1
