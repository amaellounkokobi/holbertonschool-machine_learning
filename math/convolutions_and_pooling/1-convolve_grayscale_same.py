#!/usr/bin/env python3
"""
This module contains :
A function that performs a same convolution on grayscale images

Function:
   def convolve_grayscale_same(images, kernel):
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function that performs a same convolution on grayscale images:
    ￼
    Args:
    images: is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
    ￼       m is the number of images
    ￼       h is the height in pixels of the images
    ￼       w is the width in pixels of the images

    kernel: is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
    ￼       kh is the height of the kernel
    ￼       kw is the width of the kernel

    Returns:
       A numpy.ndarray containing the convolved images
    """
    # Initialize the stride

    # Initialize values
    m = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]

    Fh = kernel.shape[0]
    Fw = kernel.shape[1]

    # Calculate the output convoluted images size
    out_w = W
    out_h = H

    # Calculate padding
    pad_H = int(Fh / 2)
    pad_W = int(Fw / 2)

    # Initialize output images

    grayscaled_imgs = np.zeros(shape=(m, out_h, out_w))

    images_pad = np.zeros((m, images.shape[1] + 2 * pad_H,
                           images.shape[2] + 2 * pad_W))

    images_pad[:, pad_H:pad_H + H, pad_W: pad_W + W] = images

    # Perform the convolution on all images
    for y in range(H):
        for x in range(W):

            filter_img = images_pad[:, y:Fh + y, x:Fw + x]
            op = np.sum(filter_img * kernel, axis=1)
            convoluted = np.sum(op, axis=1, keepdims=True)
            # add operations in all layer images
            grayscaled_imgs[:, y:y + 1, x] = convoluted

    return grayscaled_imgs
