#!/usr/bin/env python3
"""
This module contains
A function that calculates the weighted moving average of a data set

Function:
   def moving_average(data, beta):
"""
import numpy as np


def moving_average(data, beta):
    """
    Function that calculates the weighted moving average of a data set

    Args:
       data: is the list of data to calculate the moving average of

       beta: is the weight used for the moving average

    Returns:
       A list containing the moving averages of data
    """

    vt = 0
    ewma = []
    wma = []
    ewma.append(vt)

    #moving average
    for num in range(len(data)):
        vt = beta * vt + (1 - beta) * data[num]
        ewma.append(vt)

    #bias correction
    for key in range(len(ewma)):
        if key > 0:
            wma.append(ewma[key] / (1 - beta**key))

    return wma
