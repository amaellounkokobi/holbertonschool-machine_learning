#!/usr/bin/env python3
"""
This module contains :
A function that performs pooling on images

Function:
   def pool(images, kernel_shape, stride, mode='max'):
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs a convolution on images
    using multiple kernels
    ￼
    Args:

    images is a numpy.ndarray with shape (m, h, w, c)
    containing multiple images
       m is the number of images
       h is the height in pixels of the images
       w is the width in pixels of the images
       c is the number of channels in the image

    kernel_shape is a tuple of (kh, kw)
       containing the kernel shape for the pooling
       kh is the height of the kernel
       kw is the width of the kernel

    stride is a tuple of (sh, sw)
       sh is the stride for the height of the image
       sw is the stride for the width of the image

    mode indicates the type of pooling
       max indicates max pooling
       avg indicates average pooling

    Returns:
       A numpy.ndarray containing the pooled images
    """

    # Initialize values

    # Image value
    m = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    c = images.shape[3]

    # Filter value
    Fh = kernel_shape[0]
    Fw = kernel_shape[1]

    # Stride value
    Sh = stride[0]
    Sw = stride[1]

    # Calculate the output convoluted images size
    out_w = int(((W - Fw) / Sw) + 1)
    out_h = int(((H - Fh) / Sh) + 1)

    # Initialize output images
    grayscaled_imgs = np.zeros(shape=(m, out_h, out_w, c))
    images_work = images

    # Perform the convolution on all images
    for y in range(out_h):
        for x in range(out_w):
            y0 = y * Sh
            y1 = Fh + y * Sh
            x0 = x * Sw
            x1 = Fw + x * Sw
            filter_img = images[:, y0:y1, x0:x1, :]
            if mode == 'max':
                pool_max = np.max(
                    filter_img, axis=(1, 2))
                pool_reshape = np.reshape(pool_max, (m, 1, c))
            else:
                pool_av = np.average(
                    filter_img, axis=(1, 2))
                pool_reshape = np.reshape(pool_av, (m, 1, c))

            # add operations in all layer images
            grayscaled_imgs[:, y:y + 1, x, :] = pool_reshape

    return grayscaled_imgs
