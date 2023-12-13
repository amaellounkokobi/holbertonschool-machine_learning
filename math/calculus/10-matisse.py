#!/usr/bin/env python3
"""
This module contains:
- A function that calculates the derivative of a polynomial

Example:
   >>> poly = [5, 3, 0, 1]
   ... print(poly_derivative(poly))
   [3, 0, 3]

Function:
   def poly_derivative(poly):

"""

def poly_derivative(poly):
    """
    This function that calculates the derivative of a polynomial

    Args:
      poly(list): the list represents the power of x that the coefficient
    """
    if type(poly) is not list:
        return None

    poly = poly[1:]

    for index in range(len(poly)):
        coef = poly[index]

        if coef is not int:
            return None

        poly[index] = coef * (index + 1)

    if len(poly) < 1:
        return [0]

    return poly
