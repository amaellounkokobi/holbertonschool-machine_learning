#!/usr/bin/env python3

"""
This module contains :
A function that performs back propagation
over a convolutional layer of a neural network

Function:
   def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    A function that performs back propagation
    over a convolutional layer of a neural network

    Args:
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the
    unactivated output of the convolutional layer
       m is the number of examples
       h_new is the height of the output
       w_new is the width of the output
       c_new is the number of channels in the output

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
       h_prev is the height of the previous layer
       w_prev is the width of the previous layer
       c_prev is the number of channels in the previous layer

    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
     containing the kernels for the convolution
       kh is the filter height
       kw is the filter width

    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the
    biases applied to the convolution

    padding is a string that is either same or valid, indicating
    the type of padding used

    stride is a tuple of (sh, sw) containing the strides for the convolution
       sh is the stride for the height
       sw is the stride for the width

    Returns:
       the partial derivatives with respect
       to the previous layer (dA_prev),
       the kernels (dW), and
       the biases (db), respectively
    """
    # dZ shape
    m, h, w, c = dZ.shape
    
    # A_prev shape 
    m,Xh, Xw, Xc = A_prev.shape
    
    # W shape 
    Kh, Kw, c_prev, c_new = W.shape

    # stride
    Sh, Sw = stride

    # Init dW
    X = A_prev
    dWh = int(((Xh - h) / Sw) + 1)
    dWw = int(((Xw - w) / Sh) + 1)
    dW = np.zeros((dWh, dWw,c_prev,c_new))
    
    # Init dA_prv
    dA_prev = np.zeros((m,Xh,Xw,Xc))

    # Init Db 
    db = np.sum(dZ, axis=(0,1,2))

    # Padding DZ
    if padding == 'same':
        pad_H = int(np.ceil(((h - 1) * Sh - h + Kh) / 2))
        pad_W = int(np.ceil(((w - 1) * Sw - w + Kw) / 2))    
        dZ_pad = np.zeros((m, h + 2 * pad_H, w + 2 * pad_W, c))
        dZ_pad[:, pad_H:pad_H + h, pad_W: pad_W + w, :] = dZ
    else:
        dZ_pad = dZ
        
    for i in range(m):
        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    # Calculate dW
                    dz_kernel = dZ[i, y, x, ch]
                    A_feature = X[i, y*Sh :Sh * y + Kh, x * Sh :Sh* x + Kw, :]
                    dW[:, :, :, ch] +=  A_feature * dz_kernel

                    # Calculate dA_prev
                    compute = np.sum(dZ_pad[i, y, x, ch] * np.flip(W[0:Kh,0:Kw,:,ch]))
                    dA_prev[i,y,x] += compute   

    return dA_prev, dW, db
