#!/usr/bin/env python3
"""
This module contains:
- A function that calculates the integral of a polynomial

Example:
   >>> poly = [5, 3, 0, 1]
   ... print(poly_integral(poly))
   [0, 5, 1.5, 0, 0.25]

Function:
   def poly_integral(poly, C=0):

"""


def poly_integral(poly, C=0):
    """
    This function that calculates the integral of a polynomial

    Args:
       poly(list): the list represents the power of x that the coefficient
       c: integer representing the integration constant
    """

    if type(poly) is not list:
        return None
    if len(poly) < 1:
        return None
    if type(C) is not int:
        return None

    poly.insert(0,C)

    for index in range(1,len(poly)):
        if type(poly[index]) is not int:
            return None
        
        coef = poly[index]
        divid = index    
        number = coef / index 
        
        if (number % 1) == 0:
            poly[index] = int(number)
        else:
            poly[index] = number

    return poly
