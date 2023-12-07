#!/usr/bin/env python3
"""
This module contains:
- A function that performs element-wise addition, subtraction, multiplication, and division

Examples:
   >>> mat1 = np.array([[11, 22, 33], [44, 55, 66]])
   ... mat2 = np.array([[1, 2, 3], [4, 5, 6]])
   ... add, sub, mul, div = np_elementwise(mat1, mat2)
   ...print("Add:\n", add, "\nSub:\n", sub, "\nMul:\n", mul, "\nDiv:\n", div)
   Add:
    [[12 24 36]
    [48 60 72]] 
   Sub:
    [[10 20 30]
    [40 50 60]] 
   Mul:
    [[ 11  44  99]
    [176 275 396]] 
Div:
    [[11. 11. 11.]
    [11. 11. 11.]]
Functions:
   np_elementwise(mat1, mat2)

"""


def np_elementwise(mat1, mat2):
    """
    This function def np_elementwise(mat1, mat2) performs 
    element-wise addition, subtraction, multiplication, and division
    
    Args:
       mat1:Numpy n dimension array
       mat2:Numpy n dimension array
    
    Returns: 
       A tuple containing the element-wise 
       sum, difference, product, and quotient, respectively

    """
    ew_sum = mat1 + mat2
    ew_dif = mat1 - mat2
    ew_pro = mat1 * mat2
    ew_div = mat1 / mat2
    
    result_operations = (ew_sum, ew_dif, ew_pro, ew_div)

    return result_operations
