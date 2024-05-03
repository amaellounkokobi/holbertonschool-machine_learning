#!/usr/bin/env python3

"""
This module contains :
A forward propagation for a simple RNN

Function:
   def rnn(rnn_cell, X, h_0):
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    rnn Neural network
    """

    # Set initial state
    steps, samples, feature = X.shape
    hiddenState = h_0.shape[1]

    # Set initial state
    H = np.zeros((steps + 1, samples, hiddenState))
    Y = np.zeros((steps, samples, rnn_cell.Wy.shape[1]))
    h_next = h_0
    H[0] = h_next
    for step in range(steps):
        h_next, y = rnn_cell.forward(h_next, X[step])
        H[step + 1] = h_next
        Y[step] = y
    return H, Y
