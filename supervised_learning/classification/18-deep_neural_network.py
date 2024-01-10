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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l_n, layer in enumerate(layers):
            if l_n == 0:
                w_ini = np.random.randn(layers[l_n], nx) * np.sqrt(2 / nx)
                self.__weights['W{0}'.format(l_n + 1)] = w_ini
                self.__weights['b{0}'.format(l_n + 1)] = np.zeros((layer, 1))
            else:
                w_ini = np.random.randn(
                    layers[l_n], layers[l_n - 1]) * np.sqrt(
                        2 / layers[l_n - 1])
                self.__weights['W{0}'.format(l_n + 1)] = w_ini
                self.__weights['b{0}'.format(l_n + 1)] = np.zeros((layer, 1))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Updates the private attribute A1 A2
        The neuron should use a sigmoid activation function

        Args:
           X is a numpy.ndarray with shape (nx, m)
           that contains the input data

        Returns:
           the private attribute A1, A2

        """

        self.__cache['A0'] = X
        for l_n in range(self.__L):
            if l_n == 0:
                activation = np.add(np.matmul(
                    self.__weights['W1'], X), self.__weights['b1'])

                A = 1/(1 + np.exp(-activation))
                self.__cache['A{0}'.format(l_n + 1)] = A
            else:
                n_W = 'W{0}'.format(l_n + 1)
                n_b = 'b{0}'.format(l_n + 1)
                n_A = 'A{0}'.format(l_n)
                activation = np.matmul(
                    self.__weights[n_W], self.__cache[n_A]) + self.__weights[n_b])

                A = 1/(1 + np.exp(-activation))
                self.__cache['A{0}'.format(l_n + 1)] = A

        return self.__cache['A{0}'.format(self.__L)], self.__cache

    @property
    def L(self):
        """
        Getting the number of layers in the neural network.
        """

        return self.__L

    @property
    def cache(self):
        """
        Getting dictionary to hold all intermediary values of the network.
        """

        return self.__cache

    @property
    def weights(self):
        """
        Getting the weights.
        """

        return self.__weights

    def layer_positif(self, value):
        """
        Test if a value is positif
        """
        if value < 0:
            return False
        return True
