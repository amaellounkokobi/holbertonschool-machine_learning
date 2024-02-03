#!/usr/bin/env python3
"""
This module contains :
A function that perform a convolution on grayscale images
with custom padding

Function:
   def convolve_grayscale_padding(images, kernel, padding):
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that a convolution on grayscale images with
    custom padding
    ￼
    Args:
    images: is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
    ￼   m is the number of images
    ￼   h is the height in pixels of the images
    ￼   w is the width in pixels of the images

    kernel: is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
￼       kh is the height of the kernel
        kw is the width of the kernel

    padding is a tuple of (ph, pw)
       ph is the padding for the height of the image
       pw is the padding for the width of the image
       the image should be padded with 0’s

    Returns:
       A numpy.ndarray containing the convolved images
    """

    # Initialize values
    m = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]

    Fh = kernel.shape[0]
    Fw = kernel.shape[1]

    # Get padding
    pad_H = padding[0]
    pad_W = padding[1]

    # Calculate the output convoluted images size
    out_w = int(np.ceil(W - Fw + 2 * pad_W + 1))
    out_h = int(np.ceil(H - Fh + 2 * pad_H + 1))

    # Initialize output images

    grayscaled_imgs = np.zeros(shape=(m, out_h, out_w))

    images_pad = np.zeros((m, images.shape[1] + 2 * pad_H,
                           images.shape[2] + 2 * pad_W))

    images_pad[:, pad_H:pad_H + H, pad_W: pad_W + W] = images

    # Perform the convolution on all images
    for y in range(out_h):
        for x in range(out_w):
            filter_img = images_pad[:, y:Fh + y, x:Fw + x]
            op = np.sum(filter_img * kernel, axis=1)
            convoluted = np.sum(op, axis=1, keepdims=True)
            # add operations in all layer images
            grayscaled_imgs[:, y:y + 1, x] = convoluted

    return grayscaled_imgs
