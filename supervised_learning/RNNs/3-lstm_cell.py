#!/usr/bin/env python3
"""
This module contains :
A class that represent a LSTM unit

Function:
class LSTMCell():
"""
import numpy as np


class LSTMCell():
    """
    This class represent a gated LSTM unit
    """

    def __init__(self, i, h, o):
        """
        this method instanciate weight and bias of
        a LSTM unit
        """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        This method perform a forward pass for
        a LSTM unit
        """
        # compute concat hidden prev state and feature
        first_stage = np.concatenate((h_prev, x_t), axis=1)

        # compute forget gate sigmoid
        f = self.sigmoid(np.dot(first_stage, self.Wf) + self.bf)

        # compute the input gate sigmoid
        u = self.sigmoid(np.dot(first_stage, self.Wu) + self.bu)

        # compute the output gate sigmoid
        o = self.sigmoid(np.dot(first_stage, self.Wo) + self.bo)

        # compute the candidate prediction
        c_hat = np.tanh(np.dot(first_stage, self.Wc) + self.bc)

        # compute the cell state
        c_next = f * c_prev + u * c_hat

        # compute next activation
        h_next = o * np.tanh(c_next)

        # compute softmax output
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y

    def sigmoid(self, x):
        """
        This method perform a sigmoid operation
        """
        return 1 / (1 + np.exp(-x))
