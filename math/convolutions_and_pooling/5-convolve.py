#!/usr/bin/env python3
"""
This module contains :
A function that performs a convolution on images using multiple kernels
with custom padding

Function:
   def convolve(images, kernels, padding='same', stride=(1, 1)):
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
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

    kernel is a numpy.ndarray with shape (kh, kw, c, nc)
    containing the kernel for the convolution
       kh is the height of the kernel
       kw is the width of the kernel
       c is the number of channels in the image
       nc is the number of kernels

    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
       if ‘same’, performs a same convolution
       if ‘valid’, performs a valid convolution
       if a tuple:
          ph is the padding for the height of the image
          pw is the padding for the width of the image

    stride is a tuple of (sh, sw)
       sh is the stride for the height of the image
       sw is the stride for the width of the image

    Returns:
       A numpy.ndarray containing the convolved images
    """

    # Initialize values

    # Image value
    m = images.shape[0]
    H = images.shape[1]
    W = images.shape[2]
    c = images.shape[3]

    # Filter value
    Fh = kernels.shape[0]
    Fw = kernels.shape[1]
    Fn = kernels.shape[3]

    # Stride value
    Sh = stride[0]
    Sw = stride[1]

    # Padding mode
    if padding == 'same':
        out_w = W
        out_h = H

        # Calculate padding
        pad_H = int(np.ceil(((out_h - 1) * Sh - H + Fh) / 2))
        pad_W = int(np.ceil(((out_w - 1) * Sw - W + Fw) / 2))

        # Initialize output images
        grayscaled_imgs = np.zeros(shape=(m, out_h, out_w, Fn))
        images_pad = np.zeros((m, H + 2 * pad_H, W + 2 * pad_W, c))
        images_pad[:, pad_H:pad_H + H, pad_W: pad_W + W, :] = images
        images_work = images_pad

    elif padding == 'valid':
        # Calculate the output convoluted images size
        out_w = int(((W - Fw) / Sw) + 1)
        out_h = int(((H - Fh) / Sh) + 1)

        # Initialize output images
        grayscaled_imgs = np.zeros(shape=(m, out_h, out_w, Fn))
        images_work = images

    else:
        pad_H = padding[0]
        pad_W = padding[1]

        # Calculate the output convoluted images size
        out_w = int(((W + (2 * pad_W) - Fw) / Sw) + 1)
        out_h = int(((H + (2 * pad_H) - Fh) / Sh) + 1)

        # Initialize output images
        grayscaled_imgs = np.zeros(shape=(m, out_h, out_w, Fn))
        images_pad = np.zeros((m, H + 2 * pad_H, W + 2 * pad_W, c))
        images_pad[:, pad_H:pad_H + H, pad_W: pad_W + W, :] = images
        images_work = images_pad

    # adapt images for seveal kernels
    images_work = np.repeat(images_work[:, :, :, :, np.newaxis],
                            Fn, axis=len(images_work.shape))

    # Perform the convolution on all images
    for y in range(out_h):
        for x in range(out_w):
            y0 = y * Sh
            y1 = Fh + y * Sh
            x0 = x * Sw
            x1 = Fw + x * Sw
            filter_img = images_work[:, y0:y1, x0:x1, :]
            op = np.sum(filter_img * kernels, axis=1)
            conv_sum = np.sum(op, axis=2, keepdims=True)
            conv_sum_ch = np.sum(conv_sum, axis=1)
            
            # add operations in all layer images
            grayscaled_imgs[:, y:y + 1, x, :] = conv_sum_ch

    return grayscaled_imgs
