#!/usr/bin/env python3
"""
This module contains:
- A function calculates the sum  from i = 1 to n of i^2

Example:
   >>> n = 5
   ... print(summation_i_squared(n))
   55

Function:
   def summation_i_squared(n):

"""


def summation_i_squared(n):
    """
    This function calculates the sum of i^2
    from i = 1 to n equals 0

    Args:
       n(int):An integer superior to 1
    """
    start_index = 1

    if type(n) is not int and n < start_index:
        return None

    nums = [*range(start_index, n+1)]
    pow_nums = list(map(lambda num : pow(num, 2), nums))
    result = sum(pow_nums)

    return result
