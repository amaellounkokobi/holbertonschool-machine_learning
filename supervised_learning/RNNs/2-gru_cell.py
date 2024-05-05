#!/usr/bin/env python3
"""
This module contains :
A class that represent a GRU unit

Function:
class GRUCell():
"""
import numpy as np


class GRUCell():
    """
    This class represent a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        this method instanciate weight and bias of
        a GRU unit
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        This method perform a forward pass for
        a GRU unit
        """
        # compute concat hidden prev state and feature
        first_stage = np.concatenate((h_prev, x_t), axis=1)

        # compute the update gate sigmoid
        z = self.sigmoid(np.dot(first_stage, self.Wz) + self.bz)

        # compute the reset gate sigmoid
        r = self.sigmoid(np.dot(first_stage, self.Wr) + self.br)

        # compute the reset gate with H_prev + X
        reset_gate = np.concatenate((r * h_prev, x_t), axis=1)

        # compute the candidate prediction
        h_hat = np.tanh(np.dot(reset_gate, self.Wh) + self.bh)

        # compute next activation
        h_next = (1 - z) * h_prev + z * h_hat

        # compute softmax output
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y

    def sigmoid(self, x):
        """
        This method perform a sigmoid operation
        """
        return 1 / (1 + np.exp(-x))
