#!/usr/bin/env python3
"""
This module contains :
A class that defines a Deep neural network performing
binary classification

Class:
   DeepNeuralNetwork

Import:
   Numpy Library

"""
import numpy as np


class DeepNeuralNetwork():
    """
    This class defines a Deep neural network

    Attributs:


    Raises:

    """

    def __init__(self, nx, layers):
        """
        This constructor initialise class attributes:

        Args:
           nx:The number of input features to the neuron

           layers:A list representing the number of nodes
           in each layer of the network

        Raise:
           TypeError with the exception: nx must be an integer

           ValueError with the exception: nx must be a positive integer

           TypeError with the exception: layers must
           be a list of positive integers

        """
        layer_positif = np.vectorize(self.layer_positif)

        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or not layers:
            raise TypeError('layers must be a list of positive integers')
        if all(layer_positif(layers)) is False:
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l_n, layer in enumerate(layers):
            if l_n == 0:
                w_ini = np.random.randn(layers[l_n], nx) * np.sqrt(2 / nx)
                self.weights['W{0}'.format(l_n + 1)] = w_ini
                self.weights['b{0}'.format(l_n + 1)] = np.zeros((layer, 1))
            else:
                w_ini = np.random.randn(
                    layers[l_n], layers[l_n - 1]) * np.sqrt(
                        2 / layers[l_n - 1])
                self.weights['W{0}'.format(l_n + 1)] = w_ini
                self.weights['b{0}'.format(l_n + 1)] = np.zeros((layer, 1))

    def layer_positif(self, value):
        """
        Test if a value is positif
        """
        if value < 0:
            return False
        return True
