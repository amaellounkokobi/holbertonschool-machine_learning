#!/usr/bin/env python3
"""
This module contains :
A function that performs a valid convolution on grayscale images

Function:
   def convolve_grayscale_valid(images, kernel):
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images:

    Args:
    images: is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
       m is the number of images
       h is the height in pixels of the images
       w is the width in pixels of the images

    kernel: is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
       kh is the height of the kernel
       kw is the width of the kernel

    Returns:
       A numpy.ndarray containing the convolved images
    """
    # Initialize the stride
    stride = (1, 1)

    # Initialize values
    m = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]

    Fh = kernel.shape[0]
    Fw = kernel.shape[1]

    Sh = stride[0]
    Sw = stride[1]



    # Perform the convolution on all images
    for y in range(out_h):
        for x in range(out_w):
            filter_img = images[:, y:Fh + y, x:Fw + x]
            op = np.sum(filter_img * kernel, axis=1)
            convoluted = np.sum(op, axis=1, keepdims=True)
            # add operations in all layer images
            grayscaled_imgs[:, y:y + 1, x] = convoluted

    return grayscaled_imgs
