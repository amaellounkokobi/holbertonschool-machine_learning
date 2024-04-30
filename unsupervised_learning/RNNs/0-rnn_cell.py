#!/usr/bin/env python3
"""
This module contains :
A class that represents a cell of a simple RNN

Function:
class RNNCell():
"""
import numpy as np


class RNNCell():
    """
    Rnn Cell
    """
    def __init__(self, i, h, o):
        """
        RNN Init weight and bias
        """
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.random.randn(h, o)
        self.Wy = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        RNN Forward propagation
        """
        # compute next hidden state
        inputs = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(inputs, self.Wh) + self.bh)

        # compute softmax output
        y = np.dot(h_next, self.Wy.T) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
