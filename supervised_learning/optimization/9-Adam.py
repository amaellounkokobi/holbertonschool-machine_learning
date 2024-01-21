#!/usr/bin/env python3
"""
This module contains
A function that·updates·a·variable·in·place·using$
the·Adam·optimization·algorithm

Function:
def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Function that updates a variable in place using
    the Adam optimization algorithm

    Args:
       alpha: is the learning rate
       beta1: is the weight used for the first moment
       beta2: is the weight used for the second moment
       epsilon: is a small number to avoid division by zero
       var: is a numpy.ndarray containing the variable to be updated
       grad: is a numpy.ndarray containing the gradient of var
       v: is the previous first moment of var
       s: is the previous second moment of var
       t: is the time step used for bias correction
    Returns:
       the updated variable, the new first moment, and
       the new second moment, respectively
    """

    vdX = beta1 * v + (1 - beta1) * grad
    sdX = beta2 * s + (1 - beta2) * grad**2

    vdX_corr = vdX / (1 - beta1**t)
    sdX_corr = sdX / (1 - beta2**t)

    var = var - (alpha * (vdX_corr / (np.sqrt(sdX_corr) + epsilon)))

    return var, vdX, sdX
